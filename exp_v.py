import os, sys, time, math, collections, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r"C:\Users\srika\Downloads\eris_reasonin")
from eris_metric import tok, content, STOP, set_f1

torch.set_num_threads(int(os.environ.get("NT", "6")))
MODE = os.environ.get("MODE", "T")          # T = predict D\A ; D = predict D then subtract A
EPOCHS = int(os.environ.get("EPOCHS", "20"))
DIM = int(os.environ.get("DIM", "160"))
HID = int(os.environ.get("HID", "640"))
FOLDS = int(os.environ.get("FOLDS", "3"))
MAXLEN = 200
DF_MIN = int(os.environ.get("DF_MIN", "3"))
DROP = float(os.environ.get("DROP", "0.3"))
LR = float(os.environ.get("LR", "2e-3"))

tr = pd.read_csv("dataset/public/train.csv")
S = [tok(s) for s in tr.reasoning_trace]
C = [tok(s) for s in tr.canonical_restatement]
A = [set(content(s)) for s in tr.reasoning_trace]
D = [set(content(s)) for s in tr.canonical_restatement]
T = [d - a for d, a in zip(D, A)]
N = len(tr)

wc = collections.Counter()
for s in S: wc.update(s)
for c in C: wc.update(c)
itos = ["<pad>", "<unk>"] + [w for w, c in wc.most_common() if c >= 2]
stoi = {w: i for i, w in enumerate(itos)}

tgt_sets = T if MODE == "T" else D
dfc = collections.Counter()
for s in tgt_sets: dfc.update(s)
terms = [w for w, c in dfc.most_common() if c >= DF_MIN]
tix = {w: i for i, w in enumerate(terms)}
NT = len(terms)
tset = set(terms)
print(f"MODE={MODE} vocab={NT} recall_ceiling_on_T={np.mean([len(t & tset)/max(1,len(t)) for t in T]):.4f}", flush=True)

X = np.zeros((N, MAXLEN), dtype=np.int64)
for i, s in enumerate(S):
    for j, w in enumerate(s[:MAXLEN]): X[i, j] = stoi.get(w, 1)
Y = np.zeros((N, NT), dtype=np.float32)
for i, s in enumerate(tgt_sets):
    for w in s:
        if w in tix: Y[i, tix[w]] = 1.
PRIOR = Y.mean(0)


class Net(nn.Module):
    def __init__(s, V, NT, d=DIM):
        super().__init__()
        s.emb = nn.Embedding(V, d, padding_idx=0)
        nn.init.normal_(s.emb.weight, 0, d ** -0.5)
        s.c3 = nn.Conv1d(d, d, 3, padding=1)
        s.c5 = nn.Conv1d(d, d, 5, padding=2)
        s.ln = nn.LayerNorm(4 * d)
        s.f1 = nn.Linear(4 * d, HID)
        s.do = nn.Dropout(DROP)
        s.out = nn.Linear(HID, NT)
        with torch.no_grad():
            s.out.bias.copy_(torch.log(torch.tensor(np.clip(PRIOR, 1e-4, 1 - 1e-4) / (1 - np.clip(PRIOR, 1e-4, 1 - 1e-4)), dtype=torch.float32)))

    def forward(s, x):
        m = (x != 0).float().unsqueeze(-1)
        e = s.emb(x)
        h = e.transpose(1, 2)
        h = (F.gelu(s.c3(h)) + F.gelu(s.c5(h))).transpose(1, 2) * m
        den = m.sum(1).clamp(min=1)
        z = torch.cat([(e * m).sum(1) / den, (e + (m - 1) * 1e4).max(1).values,
                       h.sum(1) / den, (h + (m - 1) * 1e4).max(1).values], -1)
        return s.out(s.do(F.gelu(s.f1(s.ln(z)))))


folds = np.arange(N) % FOLDS
OOF = np.zeros((N, NT), dtype=np.float32)
t0 = time.time()
for f in range(FOLDS):
    trn = np.where(folds != f)[0]; val = np.where(folds == f)[0]
    torch.manual_seed(f)
    net = Net(len(itos), NT)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    nb = math.ceil(len(trn) / 32)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb, pct_start=0.25)
    xt = torch.from_numpy(X[trn]); yt = torch.from_numpy(Y[trn])
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(trn))
        for b in range(nb):
            ix = perm[b * 32:(b + 1) * 32]
            loss = F.binary_cross_entropy_with_logits(net(xt[ix]), yt[ix]) * NT / 20.
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        OOF[val] = torch.sigmoid(net(torch.from_numpy(X[val]))).numpy()
    print(f"  fold{f} {time.time()-t0:.0f}s", flush=True)

# mask: a predicted introduced term can never be a word already in the trace
M = np.ones((N, NT), dtype=np.float32)
for i, a in enumerate(A):
    for w in a:
        if w in tix: M[i, tix[w]] = 0.
P = OOF * M


def adaptive(p, kmax=34):
    o = np.argsort(-p); ps = p[o]; cum = np.cumsum(ps); E = float(p.sum())
    ks = np.arange(1, min(kmax, len(ps)) + 1)
    k = int(ks[int(np.argmax(2 * cum[:len(ks)] / (ks + E)))])
    return o[:k]


def ev(sel):
    return float(np.mean([set_f1(set(terms[j] for j in sel(i)), T[i]) for i in range(N)]))


print("\n--- prior-only baseline ---")
pr = PRIOR * M
print(f"  prior adaptive : {ev(lambda i: adaptive(pr[i])):.4f}")
for k in [10, 13, 16]:
    print(f"  prior fixed k={k}: {ev(lambda i, k=k: np.argsort(-pr[i])[:k]):.4f}")

print("\n--- model ---")
print(f"  adaptive-k     : {ev(lambda i: adaptive(P[i])):.4f}")
best = (0, None)
for k in [6, 8, 10, 12, 13, 14, 16, 18, 20, 24]:
    v = ev(lambda i, k=k: np.argsort(-P[i])[:k])
    if v > best[0]: best = (v, f"fixed k={k}")
    print(f"  fixed k={k:2d}     : {v:.4f}")

print("\n--- prob calibration (power) + adaptive ---")
for g in [0.5, 0.7, 1.0, 1.4, 2.0]:
    Pg = P ** g
    v = ev(lambda i, Pg=Pg: adaptive(Pg[i]))
    if v > best[0]: best = (v, f"power {g} adaptive")
    print(f"  power {g}: {v:.4f}")

print("\n--- shrink * adaptive ---")
for sh in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8]:
    Ps = np.clip(P * sh, 0, 1)
    v = ev(lambda i, Ps=Ps: adaptive(Ps[i]))
    if v > best[0]: best = (v, f"shrink {sh}")
    print(f"  shrink {sh}: {v:.4f}")

print("\n--- blend with prior ---")
for a in [0.0, 0.1, 0.2, 0.35, 0.5]:
    Pb = (1 - a) * P + a * pr
    v = ev(lambda i, Pb=Pb: adaptive(Pb[i]))
    if v > best[0]: best = (v, f"blend {a}")
    print(f"  blend {a}: {v:.4f}")

print(f"\nBEST: {best[1]} = {best[0]:.4f}   (multitask head in solution.py gave 0.2005)")
np.save(f"working/oof_terms_{MODE}.npy", OOF)
json.dump(terms, open(f"working/terms_{MODE}.json", "w"))
print("total", round(time.time() - t0), "s")
