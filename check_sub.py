import sys, re, pandas as pd, numpy as np

sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "working/submission.csv")
samp = pd.read_csv("dataset/public/sample_submission.csv")
te = pd.read_csv("dataset/public/test.csv")
bad = 0

assert list(sub.columns) == ["id", "predicted_restatement", "introduced_terms"], f"columns: {list(sub.columns)}"
assert len(sub) == 250 == len(samp), f"rows {len(sub)}"
assert set(sub.id) == set(samp.id), "id mismatch"
assert not sub.id.duplicated().any(), "dup ids"

empty = sub.predicted_restatement.isna() | (sub.predicted_restatement.astype(str).str.strip() == "")
if empty.any():
    print(f"FAIL {empty.sum()} empty predicted_restatement"); bad += 1

WORD = re.compile(r"^\w+$")
nterm, badterm, emptyterm = [], 0, 0
for s in sub.introduced_terms.fillna(""):
    ts = [t.strip() for t in str(s).split(";") if t.strip()]
    if not ts:
        emptyterm += 1
    for t in ts:
        if not WORD.match(t):
            badterm += 1
            if badterm <= 5:
                print("  BAD TERM:", repr(t))
        if t != t.lower():
            print("  NOT LOWERCASE:", repr(t))
    nterm.append(len(set(ts)))
if badterm:
    print(f"FAIL {badterm} malformed terms"); bad += 1
if emptyterm:
    print(f"WARN {emptyterm} rows with empty introduced_terms (score 0 on that item)")

plen = sub.predicted_restatement.astype(str).str.split().str.len()
print(f"pred length: mean {plen.mean():.1f} min {plen.min()} max {plen.max()}  (canonical ref ~42)")
print(f"terms/row:   mean {np.mean(nterm):.1f} min {min(nterm)} max {max(nterm)}  (true median 13)")

# guard against the 'verbatim slice of the trace' prohibition
tr = dict(zip(te.id, te.reasoning_trace))
_p = re.compile(r"[^\w\s]")
verb = 0
for i, p in zip(sub.id, sub.predicted_restatement.astype(str)):
    pt = _p.sub("", p.lower()).split()
    st = _p.sub("", str(tr[i]).lower()).split()
    if len(pt) >= 8 and " ".join(pt) in " ".join(st):
        verb += 1
print(f"predictions that are a verbatim span of their trace: {verb}/250")
if verb > 5:
    print("FAIL too many verbatim trace spans"); bad += 1

print("\nSAMPLE:")
for k in range(3):
    print(" ", sub.predicted_restatement.iloc[k][:150])
    print("   terms:", sub.introduced_terms.iloc[k][:120])
print("\nRESULT:", "PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
