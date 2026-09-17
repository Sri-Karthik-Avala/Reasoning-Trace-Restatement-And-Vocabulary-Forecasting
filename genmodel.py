import math, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_vocab(seqs, min_count=2, cap=30000):
    wc = collections.Counter()
    for s in seqs:
        wc.update(s)
    itos = list(SPECIALS) + [w for w, c in wc.most_common(cap) if c >= min_count]
    return itos, {w: i for i, w in enumerate(itos)}


def encode_src(toks, stoi, maxlen, max_oov):
    toks = toks[:maxlen]
    ids, ext, oovs = [], [], []
    for w in toks:
        i = stoi.get(w, UNK)
        ids.append(i)
        if i == UNK:
            if w not in oovs:
                if len(oovs) < max_oov:
                    oovs.append(w)
                else:
                    ext.append(UNK)
                    continue
            ext.append(len(stoi) + oovs.index(w))
        else:
            ext.append(i)
    return ids, ext, oovs


def encode_tgt(toks, stoi, oovs, maxlen):
    base = len(stoi)
    out = []
    for w in toks[:maxlen]:
        i = stoi.get(w, UNK)
        if i == UNK and w in oovs:
            i = base + oovs.index(w)
        out.append(i)
    return out


class PosEnc(nn.Module):
    def __init__(s, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        s.register_buffer("pe", pe.unsqueeze(0))

    def forward(s, x):
        return x + s.pe[:, : x.size(1)]


class DecLayer(nn.Module):
    def __init__(s, d, h, ff, p):
        super().__init__()
        s.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.l1, s.l2, s.l3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        s.do = nn.Dropout(p)

    def forward(s, x, mem, tmask, mpad, need_w=False):
        h = s.l1(x)
        a, _ = s.sa(h, h, h, attn_mask=tmask, need_weights=False)
        x = x + s.do(a)
        h = s.l2(x)
        a, w = s.ca(h, mem, mem, key_padding_mask=mpad, need_weights=need_w, average_attn_weights=True)
        x = x + s.do(a)
        x = x + s.do(s.ff(s.l3(x)))
        return x, w


class PGSeq2Seq(nn.Module):
    def __init__(s, V, d=256, h=4, ff=640, ne=3, nd=3, p=0.15, max_oov=24, n_terms=0):
        super().__init__()
        s.V, s.max_oov = V, max_oov
        s.emb = nn.Embedding(V, d, padding_idx=PAD)
        s.pos = PosEnc(d)
        s.drop = nn.Dropout(p)
        el = nn.TransformerEncoderLayer(d, h, ff, p, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(el, ne)
        s.encn = nn.LayerNorm(d)
        s.dec = nn.ModuleList([DecLayer(d, h, ff, p) for _ in range(nd)])
        s.decn = nn.LayerNorm(d)
        s.proj = nn.Linear(d, V)
        s.proj.weight = s.emb.weight
        s.pgen = nn.Linear(2 * d, 1)
        s.ctx = nn.Linear(d, d)
        s.term = nn.Linear(d, n_terms) if n_terms else None
        s.scale = math.sqrt(d)

    def encode(s, src):
        mpad = src == PAD
        e = s.drop(s.pos(s.emb(src) * s.scale))
        mem = s.encn(s.enc(e, src_key_padding_mask=mpad))
        return mem, mpad

    def term_logits(s, mem, mpad):
        m = (~mpad).float().unsqueeze(-1)
        pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1)
        return s.term(pooled)

    def decode_step(s, tgt_in, mem, mpad, src_ext, n_oov):
        B, L = tgt_in.shape
        x = s.drop(s.pos(s.emb(tgt_in) * s.scale))
        tmask = torch.triu(torch.full((L, L), float("-inf")), 1)
        w = None
        for i, lyr in enumerate(s.dec):
            x, wi = lyr(x, mem, tmask, mpad, need_w=(i == len(s.dec) - 1))
            if wi is not None:
                w = wi
        x = s.decn(x)
        vlog = s.proj(x)
        vdist = F.softmax(vlog, -1)
        cvec = torch.bmm(w, mem)
        pg = torch.sigmoid(s.pgen(torch.cat([x, s.ctx(cvec)], -1)))
        out = torch.zeros(B, L, s.V + n_oov, device=x.device, dtype=vdist.dtype)
        out[:, :, : s.V] = pg * vdist
        idx = src_ext.unsqueeze(1).expand(B, L, src_ext.size(1))
        out = out.scatter_add(2, idx, (1 - pg) * w)
        return out

    def forward(s, src, src_ext, tgt_in, n_oov):
        mem, mpad = s.encode(src)
        dist = s.decode_step(tgt_in, mem, mpad, src_ext, n_oov)
        tl = s.term_logits(mem, mpad) if s.term is not None else None
        return dist, tl


def make_batch(items, stoi, maxlen, max_oov, tgt_maxlen=88, with_tgt=True):
    B = len(items)
    Ls = max(len(x[0]) for x in items)
    src = np.zeros((B, Ls), dtype=np.int64)
    ext = np.zeros((B, Ls), dtype=np.int64)
    n_oov = max(len(x[2]) for x in items)
    for i, (ids, e, o) in enumerate(items):
        src[i, : len(ids)] = ids
        ext[i, : len(e)] = e
    return torch.from_numpy(src), torch.from_numpy(ext), n_oov
