# made by - Karthik
import sys, os, csv, re, math, time, json, collections
from pathlib import Path

T_START = time.time()
BUDGET = float(os.environ.get("ERIS_BUDGET", "4200"))

if len(sys.argv) > 2:
    public_dir = Path(sys.argv[1]); submission_out = Path(sys.argv[2])
else:
    public_dir = Path("dataset/public"); submission_out = Path("working/submission.csv")
submission_out.parent.mkdir(parents=True, exist_ok=True)

STOP = set("""the a an is are was were be been being have has had do does did will would
could should may might shall can to of in for on with at by from that this
it i me my we our you your he she they them and but or not so if then else
let get got just also about up out all more some very what which who when
where how why im ive ok okay oh um uh well like yeah yes no its as into
each than much here there now need want think know see use used using make
made go going take taken come look say new way one two first thats dont ill
cant wont their over these those such""".split())
_punc = re.compile(r"[^\w\s]")


def tok(s):
    return _punc.sub("", str(s).lower()).split()


def cont(ts):
    return set(t for t in ts if t not in STOP)


FALLBACK_TXT = ("i reviewed the request and identified the appropriate step "
                "then prepared the parameters and confirmed the result")
FALLBACK_TERMS = ""
csv.field_size_limit(10 ** 7)
_ids = []
with open(public_dir / "test.csv", "r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        _ids.append(row["id"])
try:
    _c = collections.Counter()
    with open(public_dir / "train.csv", "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            _c.update(cont(tok(row["canonical_restatement"])) - cont(tok(row["reasoning_trace"])))
    FALLBACK_TERMS = ";".join(sorted(w for w, _ in _c.most_common(13)))
except Exception:
    FALLBACK_TERMS = ""
with open(submission_out, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "predicted_restatement", "introduced_terms"])
    for i in _ids:
        w.writerow([i, FALLBACK_TXT, FALLBACK_TERMS])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
NTHREAD = int(os.environ.get("ERIS_THREADS", str(max(1, min(10, (os.cpu_count() or 4))))))
torch.set_num_threads(NTHREAD)
SEED = int(os.environ.get("ERIS_SEED", "0"))
DIM = int(os.environ.get("ERIS_DIM", "256"))
FF = int(os.environ.get("ERIS_FF", "640"))
NE = int(os.environ.get("ERIS_NE", "3"))
ND = int(os.environ.get("ERIS_ND", "3"))
HEADS = int(os.environ.get("ERIS_HEADS", "4"))
BS = int(os.environ.get("ERIS_BS", "16"))
LR = float(os.environ.get("ERIS_LR", "1.2e-3"))
MAXSRC = int(os.environ.get("ERIS_MAXSRC", "200"))
MAXTGT = int(os.environ.get("ERIS_MAXTGT", "84"))
MAXOOV = 24
DF_MIN = int(os.environ.get("ERIS_DFMIN", "3"))
BEAM = int(os.environ.get("ERIS_BEAM", "5"))
GEN_FRAC = float(os.environ.get("ERIS_GENFRAC", "0.82"))
HOLD = int(os.environ.get("ERIS_HOLD", "220"))
PAD, UNK, BOS, EOS = 0, 1, 2, 3

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
sample = pd.read_csv(public_dir / "sample_submission.csv")

tr_src = [tok(s) for s in train["reasoning_trace"]]
tr_tgt = [tok(s) for s in train["canonical_restatement"]]
te_src = [tok(s) for s in test["reasoning_trace"]]
N = len(tr_src)
tr_sc = [cont(t) for t in tr_src]
tr_tc = [cont(t) for t in tr_tgt]
te_sc = [cont(t) for t in te_src]
INTRO = [c - s for c, s in zip(tr_tc, tr_sc)]

rs = np.random.RandomState(SEED + 11)
perm = rs.permutation(N)
val_idx = np.sort(perm[:HOLD])
trn_idx = np.sort(perm[HOLD:])

wc = collections.Counter()
for i in trn_idx:
    wc.update(tr_src[i]); wc.update(tr_tgt[i])
itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + [w for w, c in wc.most_common() if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)

cwc = collections.Counter()
for i in trn_idx:
    cwc.update(tr_tgt[i])
out_words = [w for w, c in cwc.most_common() if c >= 2 and w in stoi]
OUT_IDS = np.array([0, 1, 2, 3] + [stoi[w] for w in out_words if stoi[w] > 3], dtype=np.int64)
NOUT = len(OUT_IDS)
in_out = np.zeros(len(itos), dtype=bool)
in_out[OUT_IDS] = True

dfc = collections.Counter()
for i in trn_idx:
    dfc.update(INTRO[i])
terms = [w for w, c in dfc.most_common() if c >= DF_MIN]
tix = {w: i for i, w in enumerate(terms)}
NTERM = len(terms)
term_prior = np.array([dfc[w] for w in terms], dtype=np.float32) / max(1, len(trn_idx))


def enc_src(ts):
    ts = ts[:MAXSRC]
    ids, ext, oov = [], [], []
    for w in ts:
        i = stoi.get(w, UNK)
        ids.append(i)
        if i == UNK:
            if w not in oov:
                if len(oov) < MAXOOV:
                    oov.append(w)
                else:
                    ext.append(UNK); continue
            ext.append(V + oov.index(w))
        else:
            ext.append(i)
    if not ids:
        ids, ext = [UNK], [UNK]
    return ids, ext, oov


def enc_tgt(ts, oov):
    out = []
    for w in ts[:MAXTGT]:
        i = stoi.get(w, UNK)
        if i == UNK and w in oov:
            i = V + oov.index(w)
        out.append(i)
    return out


TR = []
for i in range(N):
    a, b, o = enc_src(tr_src[i])
    t = enc_tgt(tr_tgt[i], o)
    avail = set(b)
    reach = [bool(x != UNK and ((x < V and in_out[x]) or x in avail)) for x in t] + [True]
    TR.append((a, b, o, t, reach))
TE = [enc_src(t) for t in te_src]


# ---------------- model ----------------
class PosEnc(nn.Module):
    def __init__(s, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
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


class Model(nn.Module):
    def __init__(s, V, nterm, out_ids, d=DIM, h=HEADS, ff=FF, ne=NE, nd=ND, p=0.15):
        super().__init__()
        s.V = V
        s.register_buffer("out_ids", torch.from_numpy(out_ids))
        s.emb = nn.Embedding(V, d, padding_idx=PAD)
        s.pos = PosEnc(d)
        s.drop = nn.Dropout(p)
        el = nn.TransformerEncoderLayer(d, h, ff, p, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(el, ne)
        s.encn = nn.LayerNorm(d)
        s.dec = nn.ModuleList([DecLayer(d, h, ff, p) for _ in range(nd)])
        s.decn = nn.LayerNorm(d)
        s.obias = nn.Parameter(torch.zeros(len(out_ids)))
        s.pgen = nn.Linear(2 * d, 1)
        s.ctx = nn.Linear(d, d)
        s.thead = nn.Sequential(nn.Linear(2 * d, 768), nn.GELU(), nn.Dropout(0.25), nn.Linear(768, nterm))
        s.chead = nn.Linear(2 * d, 1)
        s.sc = math.sqrt(d)
        nn.init.normal_(s.emb.weight, 0.0, d ** -0.5)
        with torch.no_grad():
            s.emb.weight[PAD].zero_()

    def encode(s, src):
        mpad = src == PAD
        e = s.drop(s.pos(s.emb(src) * s.sc))
        return s.encn(s.enc(e, src_key_padding_mask=mpad)), mpad

    def pool(s, mem, mpad):
        m = (~mpad).float().unsqueeze(-1)
        mean = (mem * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (mem + (m - 1) * 1e4).max(1).values
        return torch.cat([mean, mx], -1)

    def dec_dist(s, tin, mem, mpad, ext, n_oov):
        B, L = tin.shape
        x = s.drop(s.pos(s.emb(tin) * s.sc))
        tmask = torch.triu(torch.full((L, L), float("-inf")), 1)
        w = None
        for i, ly in enumerate(s.dec):
            x, wi = ly(x, mem, tmask, mpad, need_w=(i == len(s.dec) - 1))
            if wi is not None:
                w = wi
        x = s.decn(x)
        vd = F.softmax(F.linear(x, s.emb.weight[s.out_ids], s.obias), -1)
        cv = torch.bmm(w, mem)
        pg = torch.sigmoid(s.pgen(torch.cat([x, s.ctx(cv)], -1)))
        out = torch.zeros(B, L, s.V + n_oov)
        out[:, :, s.out_ids] = pg * vd
        idx = ext.unsqueeze(1).expand(B, L, ext.size(1))
        return out.scatter_add(2, idx, (1 - pg) * w)

    def step(s, last, pos_i, mem, mpad, ext, n_oov, cache, ew=None):
        x = s.emb(last) * s.sc + s.pos.pe[:, pos_i: pos_i + 1]
        w = None
        for i, ly in enumerate(s.dec):
            h = ly.l1(x)
            cache[i] = h if cache[i] is None else torch.cat([cache[i], h], 1)
            a, _ = ly.sa(h, cache[i], cache[i], need_weights=False)
            x = x + a
            h2 = ly.l2(x)
            a, wi = ly.ca(h2, mem, mem, key_padding_mask=mpad,
                          need_weights=(i == len(s.dec) - 1), average_attn_weights=True)
            if wi is not None:
                w = wi
            x = x + a
            x = x + ly.ff(ly.l3(x))
        x = s.decn(x)
        if ew is None:
            ew = s.emb.weight[s.out_ids]
        vd = F.softmax(F.linear(x, ew, s.obias), -1)
        cv = torch.bmm(w, mem)
        pg = torch.sigmoid(s.pgen(torch.cat([x, s.ctx(cv)], -1)))
        B = x.size(0)
        out = torch.zeros(B, 1, s.V + n_oov)
        out[:, :, s.out_ids] = pg * vd
        out = out.scatter_add(2, ext.unsqueeze(1), (1 - pg) * w)
        return out[:, 0, :]


def collate(idxs, data, intro=None):
    items = [data[i] for i in idxs]
    B = len(items)
    Ls = max(len(x[0]) for x in items)
    n_oov = max(1, max(len(x[2]) for x in items))
    src = np.zeros((B, Ls), dtype=np.int64); ext = np.zeros((B, Ls), dtype=np.int64)
    for b, it in enumerate(items):
        src[b, : len(it[0])] = it[0]; ext[b, : len(it[1])] = it[1]
    out = [torch.from_numpy(src), torch.from_numpy(ext), n_oov]
    if intro is not None:
        Lt = max(len(x[3]) for x in items) + 1
        ti = np.zeros((B, Lt), dtype=np.int64); to = np.full((B, Lt), -100, dtype=np.int64)
        ty = np.zeros((B, NTERM), dtype=np.float32); tc = np.zeros(B, dtype=np.float32)
        for b, it in enumerate(items):
            seq = [BOS] + it[3] + [EOS]
            inp = [x if x < V else UNK for x in seq[:-1]]
            ti[b, : len(inp)] = inp
            for q, (y, ok) in enumerate(zip(seq[1:], it[4])):
                if ok:
                    to[b, q] = y
            for w in intro[idxs[b]]:
                if w in tix:
                    ty[b, tix[w]] = 1.
            tc[b] = len(intro[idxs[b]])
        out += [torch.from_numpy(ti), torch.from_numpy(to), torch.from_numpy(ty), torch.from_numpy(tc)]
    return out


def make_batches(idxs, bs, shuffle=True, data=None):
    idxs = np.array(idxs)
    order = idxs[np.argsort([len(data[i][0]) for i in idxs], kind="stable")]
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        np.random.shuffle(bl)
    return bl


def train_model(seed, t_end, train_ids):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(V, NTERM, OUT_IDS).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
    bl = make_batches(train_ids, BS, False, TR)
    nb = len(bl)
    warm = max(60, int(0.10 * nb * 8))
    step = 0
    t0 = time.time()
    span = max(1.0, t_end - t0)
    ep = 0
    while True:
        if time.time() >= t_end:
            break
        m.train()
        run, nrun = 0., 0
        for bidx in make_batches(train_ids, BS, True, TR):
            frac = min(1.0, (time.time() - t0) / span)
            if step < warm:
                lr = LR * (step + 1) / warm
            else:
                lr = 1e-5 + 0.5 * (LR - 1e-5) * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            src, ext, n_oov, ti, to, ty, tc = collate(bidx, TR, INTRO)
            mem, mpad = m.encode(src)
            dist = m.dec_dist(ti, mem, mpad, ext, n_oov)
            msk = to != -100
            p = dist.gather(2, to.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            nll = -(torch.log(p + 1e-8) * msk).sum() / msk.sum()
            pooled = m.pool(mem, mpad)
            tl = m.thead(pooled)
            tloss = F.binary_cross_entropy_with_logits(tl, ty) * NTERM / 20.
            closs = F.smooth_l1_loss(m.chead(pooled).squeeze(-1), tc)
            loss = nll + 0.35 * tloss + 0.01 * closs
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); step += 1
            run += nll.item(); nrun += 1
            if step == 1:
                print(f"[seed{seed}] first-step nll {nll.item():.3f} (uniform ref {math.log(NOUT):.2f})", flush=True)
            if time.time() >= t_end:
                break
        ep += 1
        print(f"[seed{seed}] epoch {ep} nll {run/max(1,nrun):.3f} lr {lr:.2e} elapsed {time.time()-T_START:.0f}s", flush=True)
    for g in opt.param_groups:
        g["lr"] = 1e-5
    m.eval()
    return m, ep


@torch.no_grad()
def term_probs(m, data, idxs, bs=32):
    out = np.zeros((len(idxs), NTERM), dtype=np.float32)
    cnt = np.zeros(len(idxs), dtype=np.float32)
    for s0 in range(0, len(idxs), bs):
        ch = idxs[s0:s0 + bs]
        src, ext, n_oov = collate(ch, data)[:3]
        mem, mpad = m.encode(src)
        pooled = m.pool(mem, mpad)
        out[s0:s0 + len(ch)] = torch.sigmoid(m.thead(pooled)).numpy()
        cnt[s0:s0 + len(ch)] = m.chead(pooled).squeeze(-1).numpy()
    return out, cnt


@torch.no_grad()
def generate(m, data, idxs, beam=BEAM, minlen=26, maxlen=54, bs=96):
    res = {}
    for s0 in range(0, len(idxs), bs):
        ch = list(idxs[s0:s0 + bs])
        items = [data[i] for i in ch]
        B = len(ch)
        src, ext, n_oov = collate(ch, data)[:3]
        mem, mpad = m.encode(src)
        K = beam
        mem = mem.repeat_interleave(K, 0); mpad = mpad.repeat_interleave(K, 0)
        exte = ext.repeat_interleave(K, 0)
        last = torch.full((B * K, 1), BOS, dtype=torch.long)
        cache = [None] * len(m.dec)
        ew = m.emb.weight[m.out_ids]
        toks = torch.zeros(B * K, 0, dtype=torch.long)
        sc = torch.full((B, K), -1e9); sc[:, 0] = 0.; sc = sc.reshape(-1)
        done = torch.zeros(B * K, dtype=torch.bool)
        arB = torch.arange(B).unsqueeze(1)
        for st in range(maxlen):
            d = m.step(last, st, mem, mpad, exte, n_oov, cache, ew)
            lp = torch.log(d + 1e-9)
            lp[:, PAD] = -1e9; lp[:, BOS] = -1e9; lp[:, UNK] = -1e9
            if st < minlen:
                lp[:, EOS] = -1e9
            if bool(done.any()):
                lp[done] = -1e9
                lp[done, EOS] = 0.
            cand = (sc.unsqueeze(1) + lp).view(B, -1)
            top, tid = cand.topk(K, -1)
            W = lp.size(1)
            bi = torch.div(tid, W, rounding_mode="floor"); wi = tid % W
            rows = (arB * K + bi).reshape(-1)
            for i in range(len(cache)):
                cache[i] = cache[i][rows]
            flat = wi.reshape(-1)
            toks = torch.cat([toks[rows], flat.unsqueeze(1)], 1)
            done = done[rows] | (flat == EOS)
            last = torch.where(flat < V, flat, torch.full_like(flat, UNK)).unsqueeze(1)
            sc = top.reshape(-1)
            if bool(done.all()):
                break
        tl = toks.tolist()
        scl = sc.tolist()
        for b in range(B):
            oov = items[b][2]
            cands = []
            for k in range(K):
                ws = []
                for x in tl[b * K + k]:
                    if x == EOS:
                        break
                    w = itos[x] if x < V else (oov[x - V] if x - V < len(oov) else "")
                    if w:
                        ws.append(w)
                cands.append((ws, scl[b * K + k]))
            res[ch[b]] = cands
    return res


def select(cands, lenpen, mbr):
    good = [(w, s) for w, s in cands if w]
    if not good:
        return []
    if mbr and len(good) > 1:
        best, bv = good[0][0], -1.
        for w, _ in good:
            v = 0.
            for w2, _ in good:
                if w2 is not w:
                    v += rl(w, w2)
            if v > bv:
                bv, best = v, w
        return best
    return max(good, key=lambda t: t[1] / (max(1, len(t[0])) ** lenpen))[0]


def lcs(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rl(a, b):
    if not a or not b:
        return 0.
    L = lcs(a, b)
    return 0. if L == 0 else 2 * (L / len(a)) * (L / len(b)) / ((L / len(a)) + (L / len(b)))


def pick_terms(p, kmax=34):
    o = np.argsort(-p)
    ps = p[o]
    cum = np.cumsum(ps)
    E = float(p.sum())
    ks = np.arange(1, min(kmax, len(ps)) + 1)
    k = int(ks[int(np.argmax(2 * cum[: len(ks)] / (ks + E)))])
    return set(o[:k].tolist())


# ---------------- train ----------------
t_gen_end = T_START + BUDGET * GEN_FRAC
models = []
m0, ne0 = train_model(SEED, t_gen_end, trn_idx)
models.append(m0)
print(f"model0 trained {ne0} epochs at {time.time()-T_START:.0f}s", flush=True)

te_list = list(range(len(TE)))
tp, tc_ = term_probs(m0, TE, te_list)
for j, i in enumerate(te_list):
    for w in te_sc[i]:
        if w in tix:
            tp[j, tix[w]] = 0.
tg = generate(m0, TE, te_list, beam=BEAM)
print(f"test decoded t={time.time()-T_START:.0f}s", flush=True)


def write_sub(cfg, sh, a):
    rows = []
    for j, i in enumerate(te_list):
        ws = select(tg[i], cfg[0], cfg[1])[: cfg[2] or None]
        txt = " ".join(ws).strip() or FALLBACK_TXT
        p = tp[j] * sh
        if a > 0:
            for w in set(x for x in ws if x not in STOP):
                if w in tix:
                    p[tix[w]] = min(1.0, p[tix[w]] + a)
        P = sorted(terms[x] for x in pick_terms(np.clip(p, 0, 1)))
        rows.append({"id": test["id"].iloc[i], "predicted_restatement": txt,
                     "introduced_terms": ";".join(P) if P else FALLBACK_TERMS})
    s = sample[["id"]].merge(pd.DataFrame(rows), on="id", how="left")
    s["predicted_restatement"] = s["predicted_restatement"].fillna(FALLBACK_TXT)
    s["introduced_terms"] = s["introduced_terms"].fillna(FALLBACK_TERMS)
    s.to_csv(submission_out, index=False)
    return s


best_cfg, best_sh, best_a = (0.0, False, 0), 1.0, 0.0
sub = write_sub(best_cfg, best_sh, best_a)
print(f"baseline submission secured {sub.shape} t={time.time()-T_START:.0f}s", flush=True)

# holdout tuning of decode length / term blend
try:
  if time.time() - T_START > BUDGET * 0.93:
    raise RuntimeError("no time for tuning")
  val_list = list(val_idx)
  vp, vc = term_probs(m0, TR, val_list)
  for j, i in enumerate(val_list):
    for w in tr_sc[i]:
        if w in tix:
            vp[j, tix[w]] = 0.
  vg = generate(m0, TR, val_list, beam=BEAM)
  print(f"holdout decoded t={time.time()-T_START:.0f}s", flush=True)
  SELS = [(lp_, False) for lp_ in [0.0, 0.6, 1.0, 1.4]] + [(0.0, True)]
  CACHED = {s: {i: select(vg[i], *s) for i in val_list} for s in SELS}
  best_cfg, G = (0.0, False, 0), -1.
  for s in SELS:
      for cap_ in [0, 32, 36, 40, 44, 48]:
          g = float(np.mean([rl(CACHED[s][i][:cap_] if cap_ else CACHED[s][i], tr_tgt[i]) for i in val_list]))
          if g > G:
              G, best_cfg = g, (s[0], s[1], cap_)
  print(f"holdout G={G:.4f} cfg={best_cfg} predlen="
        f"{np.mean([len(CACHED[(best_cfg[0], best_cfg[1])][i][:best_cfg[2]] if best_cfg[2] else CACHED[(best_cfg[0], best_cfg[1])][i]) for i in val_list]):.1f}"
        f" t={time.time()-T_START:.0f}s", flush=True)

  VTXT = {i: set(w for w in select(vg[i], best_cfg[0], best_cfg[1])[: best_cfg[2] or None] if w not in STOP) for i in val_list}
  for i in val_list[:3]:
      print("  PRED:", " ".join(select(vg[i], best_cfg[0], best_cfg[1])[: best_cfg[2] or None]), flush=True)
      print("  REF :", " ".join(tr_tgt[i]), flush=True)
  best_sh, best_a, best_v = 1.0, 0.0, -1.
  for a in [0.0, 0.05, 0.1, 0.18, 0.28]:
      for sh in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
          sc = []
          for j, i in enumerate(val_list):
              p = vp[j] * sh
              if a > 0:
                  for w in VTXT[i]:
                      if w in tix:
                          p[tix[w]] = min(1.0, p[tix[w]] + a)
              P = set(terms[x] for x in pick_terms(np.clip(p, 0, 1)))
              T = INTRO[i]
              sc.append(0. if (not P or not T) else 2 * len(P & T) / (len(P) + len(T)))
          v = float(np.mean(sc))
          if v > best_v:
              best_v, best_sh, best_a = v, sh, a
  print(f"holdout V={best_v:.4f} shrink={best_sh} alpha={best_a} t={time.time()-T_START:.0f}s", flush=True)
  print(f"holdout TSS={0.6*G + 0.4*best_v:.4f}", flush=True)
except Exception as _e:
  print("tuning skipped:", repr(_e), flush=True)

sub = write_sub(best_cfg, best_sh, best_a)
print("wrote", submission_out, sub.shape, f"total {time.time()-T_START:.0f}s", flush=True)
