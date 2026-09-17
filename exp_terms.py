import os, sys, time, math, collections, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r"C:\Users\srika\Downloads\eris_reasonin")
from eris_metric import tok, content, STOP, set_f1

torch.set_num_threads(int(os.environ.get("NT", "10")))
SEED = 0
MAXLEN = int(os.environ.get("MAXLEN", "256"))
DF_MIN = int(os.environ.get("DF_MIN", "3"))
EPOCHS = int(os.environ.get("EPOCHS", "24"))
DIM = int(os.environ.get("DIM", "224"))
LR = float(os.environ.get("LR", "2e-3"))

tr = pd.read_csv("dataset/public/train.csv")
traces = [tok(s) for s in tr.reasoning_trace]
canons = [tok(s) for s in tr.canonical_restatement]
tcs = [set(w for w in t if w not in STOP) for t in traces]
ccs = [set(w for w in c if w not in STOP) for c in canons]
TT = [c - t for c, t in zip(ccs, tcs)]
N = len(tr)

wc = collections.Counter()
for t in traces: wc.update(t)
for c in canons: wc.update(c)
itos = ["<pad>", "<unk>"] + [w for w, c in wc.most_common() if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}
print("enc vocab", len(itos))

dfc = collections.Counter()
for s in TT: dfc.update(s)
terms = [w for w, c in dfc.most_common() if c >= DF_MIN]
tidx = {w: i for i, w in enumerate(terms)}
NT = len(terms)
print("term vocab", NT, "mean recall ceiling",
      np.mean([len(s & set(terms)) / max(1, len(s)) for s in TT]).round(4))

X = np.zeros((N, MAXLEN), dtype=np.int64)
XL = np.zeros(N, dtype=np.int64)
for i, t in enumerate(traces):
    t = t[:MAXLEN]
    XL[i] = len(t)
    for j, w in enumerate(t): X[i, j] = stoi.get(w, 1)
Y = np.zeros((N, NT), dtype=np.float32)
for i, s in enumerate(TT):
    for w in s:
        if w in tidx: Y[i, tidx[w]] = 1.
# mask: candidate terms present in trace can never be in T
M = np.ones((N, NT), dtype=np.float32)
for i, s in enumerate(tcs):
    for w in s:
        if w in tidx: M[i, tidx[w]] = 0.


class TermNet(nn.Module):
    def __init__(s, V, NT, d=DIM):
        super().__init__()
        s.emb = nn.Embedding(V, d, padding_idx=0)
        s.c3 = nn.Conv1d(d, d, 3, padding=1)
        s.c5 = nn.Conv1d(d, d, 5, padding=2)
        s.ln = nn.LayerNorm(4 * d)
        s.fc1 = nn.Linear(4 * d, 768)
        s.do = nn.Dropout(0.3)
        s.out = nn.Linear(768, NT)
        s.cnt = nn.Linear(768, 1)

    def forward(s, x):
        m = (x != 0).float().unsqueeze(-1)
        e = s.emb(x)
        h = e.transpose(1, 2)
        h = F.gelu(s.c3(h)) + F.gelu(s.c5(h))
        h = h.transpose(1, 2) * m
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        cmean = h.sum(1) / m.sum(1).clamp(min=1)
        cmax = (h + (m - 1) * 1e4).max(1).values
        emax = (e + (m - 1) * 1e4).max(1).values
        z = torch.cat([mean, emax, cmean, cmax], -1)
        z = s.do(F.gelu(s.fc1(s.ln(z))))
        return s.out(z), s.cnt(z).squeeze(-1)


def pick(p, k_max=40):
    order = np.argsort(-p)
    ps = p[order]
    cum = np.cumsum(ps)
    Ehat = p.sum()
    ks = np.arange(1, min(k_max, len(ps)) + 1)
    ef1 = 2 * cum[:len(ks)] / (ks + Ehat)
    k = int(ks[np.argmax(ef1)])
    return order[:k]


def run_fold(fold, folds, epochs=EPOCHS, verbose=False):
    trn = np.where(folds != fold)[0]
    val = np.where(folds == fold)[0]
    torch.manual_seed(SEED + fold)
    net = TermNet(len(itos), NT)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    nb = math.ceil(len(trn) / 32)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * nb, pct_start=0.25)
    xt = torch.from_numpy(X[trn]); yt = torch.from_numpy(Y[trn])
    ct = torch.from_numpy(np.array([len(s) for s in TT], dtype=np.float32)[trn])
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(trn))
        tot = 0.
        for b in range(nb):
            ix = perm[b * 32:(b + 1) * 32]
            lg, cn = net(xt[ix])
            loss = F.binary_cross_entropy_with_logits(lg, yt[ix]) * NT / 20.
            loss = loss + 0.02 * F.smooth_l1_loss(cn, ct[ix])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item()
        if verbose and (ep % 4 == 3 or ep == epochs - 1):
            print(f"   ep{ep} loss {tot/nb:.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(X[val]))[0]).numpy()
    P = P * M[val]
    return val, P


folds = np.arange(N) % 5
OOF = np.zeros((N, NT), dtype=np.float32)
t0 = time.time()
for f in range(5):
    v, p = run_fold(f, folds, verbose=(f == 0))
    OOF[v] = p
    print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)

vs = []
for i in range(N):
    sel = pick(OOF[i])
    vs.append(set_f1(set(terms[j] for j in sel), TT[i]))
print(f"\nADAPTIVE-k  V = {np.mean(vs):.4f}")
for K in [8, 10, 12, 13, 14, 16, 18, 20]:
    vs = [set_f1(set(terms[j] for j in np.argsort(-OOF[i])[:K]), TT[i]) for i in range(N)]
    print(f"  fixed k={K:2d}  V = {np.mean(vs):.4f}")
np.save("working/oof_terms.npy", OOF)
json.dump(terms, open("working/terms.json", "w"))
print("total", time.time() - t0)
