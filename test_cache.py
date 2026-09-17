import sys, torch
sys.argv = ["solution.py", "dataset/public", "working/_tmp_test.csv"]
src = open("solution.py", encoding="utf-8").read()
head = src.split("def collate(")[0]
g = {"__name__": "solcheck"}
exec(compile(head, "solution_head", "exec"), g)

Model, V, NTERM = g["Model"], g["V"], g["NTERM"]
PAD, BOS, UNK = g["PAD"], g["BOS"], g["UNK"]
print("V", V, "NOUT", g["NOUT"], "NTERM", NTERM)
torch.manual_seed(0)
m = Model(V, NTERM, g["OUT_IDS"])
m.eval()
print("params", round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M")
rmask = [r for it in g["TR"][:400] for r in it[4][:-1]]
print("target reachability", round(sum(rmask) / len(rmask), 4))

B, Ls, n_oov = 3, 17, 2
srct = torch.randint(4, V, (B, Ls))
srct[0, -4:] = PAD
ext = srct.clone()
ext[1, 2] = V + 0
ext[2, 5] = V + 1
tgt = torch.randint(4, V, (B, 6))
tgt[:, 0] = BOS

with torch.no_grad():
    mem, mpad = m.encode(srct)
    full = m.dec_dist(tgt, mem, mpad, ext, n_oov)
    cache = [None] * len(m.dec)
    steps = []
    for t in range(tgt.size(1)):
        steps.append(m.step(tgt[:, t:t + 1], t, mem, mpad, ext, n_oov, cache))
    inc = torch.stack(steps, 1)

d = (full - inc).abs().max().item()
print("max abs diff full-vs-cached:", d)
print("row sums (full, inc):", full.sum(-1).mean().item(), inc.sum(-1).mean().item())
assert d < 2e-5, f"KV cache mismatch {d}"
print("OK: incremental decoding matches teacher-forced decoding")
