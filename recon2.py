import pandas as pd, numpy as np, collections, sys, random
sys.path.insert(0, r"C:\Users\srika\Downloads\eris_reasonin")
from eris_metric import tok, content, cf, STOP, set_f1, rouge_l_f1

tr = pd.read_csv("dataset/public/train.csv")
tt = [tok(s) for s in tr.reasoning_trace]
ct = [tok(s) for s in tr.canonical_restatement]
tc = [set(content(s)) for s in tr.reasoning_trace]
cc = [set(content(s)) for s in tr.canonical_restatement]
TT = [c - t for c, t in zip(cc, tc)]
N = len(tr)
rs = np.random.RandomState(0)
samp = rs.choice(N, 300, replace=False)

print("=== structure ===")
starts = collections.Counter(" ".join(c[:2]) for c in ct)
print("top starts:", starts.most_common(12))
print("frac starting with 'i':", np.mean([c[0] == "i" for c in ct]).round(3))
sentcnt = [s.count(".") + s.count("!") + s.count("?") for s in tr.canonical_restatement]
print("sentences per canon:", pd.Series(sentcnt).describe().round(2).to_dict())

print("\n=== constant-prediction baseline ===")
cand = rs.choice(N, 150, replace=False)
best, bestsc = None, -1
for i in cand:
    sc = np.mean([rouge_l_f1(ct[i], ct[j]) for j in samp if j != i])
    if sc > bestsc: bestsc, best = sc, i
print(f"best single train restatement as constant: G={bestsc:.4f}")
print("  ->", tr.canonical_restatement.iloc[best][:220])

print("\n=== retrieval baseline (nearest train trace by content jaccard) ===")
def jac(a, b):
    if not a or not b: return 0.
    return len(a & b) / len(a | b)
sc = []
for i in samp:
    bj, bk = -1, -1
    for j in range(N):
        if j == i: continue
        v = jac(tc[i], tc[j])
        if v > bj: bj, bk = v, j
    sc.append(rouge_l_f1(ct[bk], ct[i]))
print(f"1-NN retrieved restatement: G={np.mean(sc):.4f}")

print("\n=== ORACLE ceilings ===")
# oracle 1: exact canonical content set, ordered as canonical -> trivially 1.0. skip.
# oracle 2: bag of canonical tokens shuffled (order destroyed)
sc = []
for i in samp:
    p = list(ct[i]); random.Random(i).shuffle(p)
    sc.append(rouge_l_f1(p, ct[i]))
print(f"oracle tokens, shuffled order: G={np.mean(sc):.4f}")
# oracle 3: trace-tokens-in-canon (extractive), already 0.378
# oracle 4: extractive + oracle introduced terms appended in canon order at end
sc = []
for i in samp:
    keep = [x for x in tt[i] if x in set(ct[i])]
    p = keep + [x for x in ct[i] if x in TT[i]]
    sc.append(rouge_l_f1(p, ct[i]))
print(f"extractive + oracle terms appended: G={np.mean(sc):.4f}")
# oracle 5: canonical tokens but only those predictable... -> content only
sc = []
for i in samp:
    p = [x for x in ct[i] if x not in STOP]
    sc.append(rouge_l_f1(p, ct[i]))
print(f"oracle canon CONTENT tokens only (stopwords dropped): G={np.mean(sc):.4f}")
sc = []
for i in samp:
    p = [x for x in ct[i] if x in STOP]
    sc.append(rouge_l_f1(p, ct[i]))
print(f"oracle canon STOPWORDS only: G={np.mean(sc):.4f}")

print("\n=== how much of canon content comes from trace, and in trace order? ===")
frac_in_trace = np.mean([len(c & t) / max(1, len(c)) for c, t in zip(cc, tc)])
print("frac canon content present in trace:", round(frac_in_trace, 3))
# order preservation: for canon content words also in trace, is canon order ~ trace first-occurrence order?
from scipy.stats import spearmanr
rhos = []
for i in samp:
    pos_t = {}
    for k, w in enumerate(tt[i]):
        pos_t.setdefault(w, k)
    seq = [(k, pos_t[w]) for k, w in enumerate(ct[i]) if w in pos_t and w not in STOP]
    if len(seq) >= 4:
        r = spearmanr([a for a, b in seq], [b for a, b in seq]).statistic
        if not np.isnan(r): rhos.append(r)
print("spearman(canon order, trace first-occ order) for shared content:", round(float(np.mean(rhos)), 3), "n=", len(rhos))

print("\n=== V ceiling with a learned ranker: oracle-k on global freq, per-item masked ===")
dfc = collections.Counter()
for s in TT: dfc.update(s)
rank = {w: i for i, (w, _) in enumerate(dfc.most_common())}
for K in [8, 10, 12, 14, 16, 18, 22]:
    v = []
    for i in range(N):
        cands = [w for w, _ in dfc.most_common(200) if w not in tc[i]][:K]
        v.append(set_f1(set(cands), TT[i]))
    print(f"  masked global top{K}: V={np.mean(v):.4f}")

print("\n=== upper bound: if we could rank perfectly within vocab>=5 ===")
vocab5 = set(w for w, c in dfc.items() if c >= 5)
for K in [10, 13, 16]:
    v = []
    for i in range(N):
        hit = list(TT[i] & vocab5)[:K]
        v.append(set_f1(set(hit), TT[i]))
    print(f"  oracle top{K} from vocab>=5: V={np.mean(v):.4f}")
