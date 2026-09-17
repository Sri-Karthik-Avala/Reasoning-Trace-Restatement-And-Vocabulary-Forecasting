import os, sys, time, math, json, collections
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r"C:\Users\srika\Downloads\eris_reasonin")
from eris_metric import tok, STOP, set_f1, rouge_l_f1
from genmodel import (PAD, UNK, BOS, EOS, build_vocab, encode_src, encode_tgt, PGSeq2Seq)

torch.set_num_threads(int(os.environ.get("NT", "10")))
MAXSRC = int(os.environ.get("MAXSRC", "200"))
MAXTGT = int(os.environ.get("MAXTGT", "84"))
MAXOOV = 24
DIM = int(os.environ.get("DIM", "256"))
NE = int(os.environ.get("NE", "3"))
ND = int(os.environ.get("ND", "3"))
FF = int(os.environ.get("FF", "640"))
BS = int(os.environ.get("BS", "16"))
LR = float(os.environ.get("LR", "1.2e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
HOLD = int(os.environ.get("HOLD", "300"))
BEAM = int(os.environ.get("BEAM", "4"))
LENPEN = float(os.environ.get("LENPEN", "1.0"))
SEED = int(os.environ.get("SEED", "0"))

tr = pd.read_csv("dataset/public/train.csv")
traces = [tok(s) for s in tr.reasoning_trace]
canons = [tok(s) for s in tr.canonical_restatement]
N = len(tr)
rs = np.random.RandomState(7)
perm = rs.permutation(N)
val_idx = perm[:HOLD]
trn_idx = perm[HOLD:]

itos, stoi = build_vocab([traces[i] for i in trn_idx] + [canons[i] for i in trn_idx], min_count=2)
V = len(itos)
print("vocab", V, flush=True)

tcs = [set(w for w in t if w not in STOP) for t in traces]
ccs = [set(w for w in c if w not in STOP) for c in canons]
TT = [c - t for c, t in zip(ccs, tcs)]
dfc = collections.Counter()
for i in trn_idx: dfc.update(TT[i])
terms = [w for w, c in dfc.most_common() if c >= 3]
tix = {w: i for i, w in enumerate(terms)}
NTERM = len(terms)
print("terms", NTERM, flush=True)


def prep(i):
    ids, ext, oovs = encode_src(traces[i], stoi, MAXSRC, MAXOOV)
    tgt = encode_tgt(canons[i], stoi, oovs, MAXTGT)
    return ids, ext, oovs, tgt


DATA = [prep(i) for i in range(N)]


def collate(idxs):
    items = [DATA[i] for i in idxs]
    B = len(items)
    Ls = max(len(x[0]) for x in items)
    Lt = max(len(x[3]) for x in items) + 1
    n_oov = max(1, max(len(x[2]) for x in items))
    src = np.zeros((B, Ls), dtype=np.int64)
    ext = np.zeros((B, Ls), dtype=np.int64)
    ti = np.zeros((B, Lt), dtype=np.int64)
    to = np.full((B, Lt), -100, dtype=np.int64)
    ty = np.zeros((B, NTERM), dtype=np.float32)
    for b, (ids, e, o, t) in enumerate(items):
        src[b, : len(ids)] = ids
        ext[b, : len(e)] = e
        seq = [BOS] + t + [EOS]
        inp = [x if x < V else UNK for x in seq[:-1]]
        ti[b, : len(inp)] = inp
        to[b, : len(seq) - 1] = seq[1:]
        for w in TT[idxs[b]]:
            if w in tix: ty[b, tix[w]] = 1.
    return (torch.from_numpy(src), torch.from_numpy(ext), torch.from_numpy(ti),
            torch.from_numpy(to), torch.from_numpy(ty), n_oov)


def batches(idxs, bs, shuffle=True):
    idxs = np.array(idxs)
    order = idxs[np.argsort([len(DATA[i][0]) for i in idxs], kind="stable")]
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        np.random.shuffle(bl)
    return bl


torch.manual_seed(SEED)
model = PGSeq2Seq(V, DIM, 4, FF, NE, ND, 0.15, MAXOOV, NTERM)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
nb = len(batches(trn_idx, BS, False))
sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb, pct_start=0.12)

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    tot = nt = 0.
    for bidx in batches(trn_idx, BS):
        src, ext, ti, to, ty, n_oov = collate(bidx)
        dist, tl = model(src, ext, ti, n_oov)
        mask = to != -100
        tgt = to.clamp(min=0)
        p = dist.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        nll = -(torch.log(p + 1e-8) * mask).sum() / mask.sum()
        tloss = F.binary_cross_entropy_with_logits(tl, ty) * NTERM / 20.
        loss = nll + 0.35 * tloss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        tot += nll.item(); nt += 1
    if ep % 3 == 2 or ep == EPOCHS - 1:
        print(f"ep{ep} nll {tot/nt:.4f}  {time.time()-t0:.0f}s", flush=True)
TRAIN_T = time.time() - t0
print("train seconds", TRAIN_T, flush=True)


@torch.no_grad()
def decode(idxs, beam=BEAM, lenpen=LENPEN, minlen=26, maxlen=64, bs=32):
    model.eval()
    out = {}
    for s0 in range(0, len(idxs), bs):
        chunk = idxs[s0:s0 + bs]
        items = [DATA[i] for i in chunk]
        B = len(items)
        Ls = max(len(x[0]) for x in items)
        n_oov = max(1, max(len(x[2]) for x in items))
        src = np.zeros((B, Ls), dtype=np.int64); ext = np.zeros((B, Ls), dtype=np.int64)
        for b, (ids, e, o, t) in enumerate(items):
            src[b, : len(ids)] = ids; ext[b, : len(e)] = e
        src = torch.from_numpy(src); ext = torch.from_numpy(ext)
        mem, mpad = model.encode(src)
        K = beam
        mem = mem.repeat_interleave(K, 0); mpad = mpad.repeat_interleave(K, 0)
        exte = ext.repeat_interleave(K, 0)
        seqs = torch.full((B * K, 1), BOS, dtype=torch.long)
        gen = [[] for _ in range(B * K)]
        scores = torch.full((B, K), -1e9); scores[:, 0] = 0.; scores = scores.view(-1)
        done = torch.zeros(B * K, dtype=torch.bool)
        for step in range(maxlen):
            dist = model.decode_step(seqs, mem, mpad, exte, n_oov)[:, -1, :]
            lp = torch.log(dist + 1e-9)
            if step < minlen:
                lp[:, EOS] = -1e9
            lp[:, PAD] = -1e9; lp[:, BOS] = -1e9; lp[:, UNK] = -1e9
            lp = torch.where(done.unsqueeze(1), torch.full_like(lp, -1e9), lp)
            if done.any():
                lp[done, EOS] = 0.
            cand = scores.unsqueeze(1) + lp
            cand = cand.view(B, K * cand.size(1))
            top, ti_ = cand.topk(K, -1)
            Vt = lp.size(1)
            bi = ti_ // Vt; wi = ti_ % Vt
            newseq, newgen, newdone = [], [], []
            for b in range(B):
                for k in range(K):
                    src_row = b * K + int(bi[b, k])
                    newseq.append(seqs[src_row])
                    g = list(gen[src_row])
                    w = int(wi[b, k])
                    d = bool(done[src_row])
                    if not d:
                        if w == EOS: d = True
                        else: g.append(w)
                    newgen.append(g); newdone.append(d)
            seqs = torch.stack([torch.cat([s, torch.tensor([min(int(w), V - 1) if int(w) < V else UNK])])
                                for s, w in zip(newseq, wi.reshape(-1))])
            gen = newgen; done = torch.tensor(newdone); scores = top.reshape(-1)
            if done.all(): break
        norm = scores.view(B, K) / (torch.tensor([[max(1, len(gen[b * K + k])) for k in range(K)] for b in range(B)]).float() ** lenpen)
        best = norm.argmax(1)
        for b in range(B):
            g = gen[b * K + int(best[b])]
            oovs = items[b][2]
            ws = [itos[x] if x < V else (oovs[x - V] if x - V < len(oovs) else "") for x in g]
            out[chunk[b]] = " ".join([w for w in ws if w])
    return out


t1 = time.time()
preds = decode(list(val_idx))
print("decode seconds", time.time() - t1, flush=True)
G = np.mean([rouge_l_f1(tok(preds[i]), canons[i]) for i in val_idx])
print(f"\nHOLDOUT G = {G:.4f}   mean pred len {np.mean([len(preds[i].split()) for i in val_idx]):.1f}", flush=True)
for i in val_idx[:4]:
    print("\nPRED:", preds[i])
    print("REF :", " ".join(canons[i]))

# term head V
model.eval()
with torch.no_grad():
    vs = []
    for s0 in range(0, len(val_idx), 32):
        ch = val_idx[s0:s0 + 32]
        src, ext, ti, to, ty, n_oov = collate(ch)
        mem, mpad = model.encode(src)
        p = torch.sigmoid(model.term_logits(mem, mpad)).numpy()
        for b, i in enumerate(ch):
            pp = p[b].copy()
            for w in tcs[i]:
                if w in tix: pp[tix[w]] = 0.
            order = np.argsort(-pp); ps = pp[order]; cum = np.cumsum(ps); E = pp.sum()
            ks = np.arange(1, 41)
            k = int(ks[np.argmax(2 * cum[:40] / (ks + E))])
            vs.append(set_f1(set(terms[j] for j in order[:k]), TT[i]))
print(f"\nHOLDOUT V (multitask head) = {np.mean(vs):.4f}")
print(f"TSS estimate = {0.6*G + 0.4*np.mean(vs):.4f}")
json.dump({str(k): v for k, v in preds.items()}, open("working/gen_preds.json", "w"))
