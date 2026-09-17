import pandas as pd, numpy as np, collections, sys
sys.path.insert(0, r"C:\Users\srika\Downloads\eris_reasonin")
from eris_metric import tok, content, cf, true_terms, STOP, set_f1

tr = pd.read_csv("dataset/public/train.csv")
te = pd.read_csv("dataset/public/test.csv")
ss = pd.read_csv("dataset/public/sample_submission.csv")
print("shapes", tr.shape, te.shape, ss.shape)
print("dupe ids", tr.id.duplicated().sum(), te.id.duplicated().sum(), "overlap", len(set(tr.id) & set(te.id)))
print("dupe traces train", tr.reasoning_trace.duplicated().sum(), "test", te.reasoning_trace.duplicated().sum())
print("trace overlap tr/te", len(set(tr.reasoning_trace) & set(te.reasoning_trace)))

tr["ttok"] = tr.reasoning_trace.map(tok)
tr["ctok"] = tr.canonical_restatement.map(tok)
tr["tc"] = tr.reasoning_trace.map(lambda s: set(content(s)))
tr["cc"] = tr.canonical_restatement.map(lambda s: set(content(s)))
tr["TT"] = [c - t for c, t in zip(tr.cc, tr.tc)]
te["ttok"] = te.reasoning_trace.map(tok)

print("\n-- lengths (tokens) --")
print("trace ", pd.Series([len(x) for x in tr.ttok]).describe().round(1).to_dict())
print("canon ", pd.Series([len(x) for x in tr.ctok]).describe().round(1).to_dict())
print("test trace", pd.Series([len(x) for x in te.ttok]).describe().round(1).to_dict())
print("|T| ", pd.Series([len(x) for x in tr["TT"]]).describe().round(1).to_dict())
print("|canon content|", pd.Series([len(x) for x in tr.cc]).describe().round(1).to_dict())
print("frac canon content that is introduced", np.mean([len(t) / max(1, len(c)) for t, c in zip(tr["TT"], tr.cc)]).round(3))
print("frac canon tokens that are stopwords", np.mean([sum(1 for x in c if x in STOP) / max(1, len(c)) for c in tr.ctok]).round(3))

# pairwise T overlap
rs = np.random.RandomState(0)
idx = rs.randint(0, len(tr), (2000, 2))
print("mean pairwise T setF1", np.mean([set_f1(tr["TT"].iloc[a], tr["TT"].iloc[b]) for a, b in idx if a != b]).round(4))

print("\n-- top 60 introduced terms (doc freq over 2000) --")
df = collections.Counter()
for s in tr["TT"]: df.update(s)
for w, c in df.most_common(60): print(f"  {w:16s} {c:5d} {c/len(tr):.3f}")
print("vocab of introduced terms:", len(df), " >=5:", sum(1 for v in df.values() if v >= 5), " >=20:", sum(1 for v in df.values() if v >= 20))

# how much of T is covered by top-K global list, after masking trace-present
for K in [10, 13, 15, 20, 30, 50]:
    top = [w for w, _ in df.most_common(K)]
    sc = np.mean([set_f1(set(top), T) for T in tr["TT"]])
    sc2 = np.mean([set_f1(set(top) - tc, T) for tc, T in zip(tr.tc, tr["TT"])])
    print(f"global top{K}: V={sc:.4f}   masked-by-trace V={sc2:.4f}")

# recall ceiling: what frac of T tokens are in top-N vocab
for N in [200, 500, 1000, 2000, 4000, 8000]:
    top = set(w for w, _ in df.most_common(N))
    rec = np.mean([len(T & top) / max(1, len(T)) for T in tr["TT"]])
    print(f"vocab top{N}: mean recall of T = {rec:.4f}")

print("\n-- baseline G --")
def firstn(t, n=42):
    return " ".join(tok(t)[:n])
print("first42 of trace:", np.mean([cf(firstn(t, 42), c) for t, c in zip(tr.reasoning_trace, tr.canonical_restatement)]).round(4))
print("first30 of trace:", np.mean([cf(firstn(t, 30), c) for t, c in zip(tr.reasoning_trace, tr.canonical_restatement)]).round(4))
print("full trace:", np.mean([cf(t, c) for t, c in zip(tr.reasoning_trace, tr.canonical_restatement)]).round(4))
# nearest-neighbour-free: fixed generic canonical
med = tr.canonical_restatement.iloc[0]
print("\n-- oracle extractive: trace tokens (in order) that occur in canon --")
def orac(tt, cset):
    return " ".join([x for x in tt if x in cset])
oc = [orac(tt, set(ct)) for tt, ct in zip(tr.ttok, tr.ctok)]
print("oracle-extract G:", np.mean([cf(p, c) for p, c in zip(oc, tr.canonical_restatement)]).round(4))
print("  mean len", np.mean([len(p.split()) for p in oc]).round(1))
