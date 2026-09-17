import sys, os, time, math, json
import numpy as np, torch, torch.nn.functional as F

sys.argv = ["solution.py", "dataset/public", "working/_lab_tmp.csv"]
src = open("solution.py", encoding="utf-8").read()
head = src.split("# ---------------- train ----------------")[0]
g = {"__name__": "lab"}
exec(compile(head, "solhead", "exec"), g)

Model, V, NTERM, OUT_IDS = g["Model"], g["V"], g["NTERM"], g["OUT_IDS"]
TR, val_idx, tr_tgt = g["TR"], g["val_idx"], g["tr_tgt"]
generate, select, rl, tok = g["generate"], g["select"], g["rl"], g["tok"]
PAD, UNK, BOS, EOS = g["PAD"], g["UNK"], g["BOS"], g["EOS"]
itos, collate = g["itos"], g["collate"]

m = Model(V, NTERM, OUT_IDS)
m.load_state_dict(torch.load("working/model.pt", map_location="cpu"))
m.eval()
torch.set_num_threads(int(os.environ.get("NT", "6")))
val = list(val_idx)
print(f"loaded model, holdout {len(val)}", flush=True)


def score(sel_map):
    return float(np.mean([rl(sel_map[i], tr_tgt[i]) for i in val]))


t0 = time.time()
print("\n=== 1. beam sweep (baseline reproduction) ===", flush=True)
for beam, minl, maxl in [(5, 26, 54), (5, 30, 60), (5, 34, 64), (8, 30, 60), (3, 26, 54)]:
    gg = generate(m, TR, val, beam=beam, minlen=minl, maxlen=maxl)
    for lp in [0.0, 0.7, 1.0]:
        s = score({i: select(gg[i], lp, False) for i in val})
        L = np.mean([len(select(gg[i], lp, False)) for i in val])
        print(f"  beam{beam} min{minl} max{maxl} lenpen{lp}: G={s:.4f} len={L:.1f}", flush=True)
    s = score({i: select(gg[i], 0.0, True) for i in val})
    print(f"  beam{beam} min{minl} max{maxl} MBR      : G={s:.4f}", flush=True)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)


@torch.no_grad()
def sample_gen(m, data, idxs, nsamp=12, temp=0.9, topk=40, maxlen=60, minlen=26, bs=24, seed=0):
    torch.manual_seed(seed)
    res = {}
    for s0 in range(0, len(idxs), bs):
        ch = list(idxs[s0:s0 + bs])
        items = [data[i] for i in ch]
        B = len(ch)
        srct, ext, n_oov = collate(ch, data)[:3]
        mem, mpad = m.encode(srct)
        K = nsamp
        mem = mem.repeat_interleave(K, 0); mpad = mpad.repeat_interleave(K, 0)
        exte = ext.repeat_interleave(K, 0)
        last = torch.full((B * K, 1), BOS, dtype=torch.long)
        cache = [None] * len(m.dec)
        ew = m.emb.weight[m.out_ids]
        toks = torch.zeros(B * K, 0, dtype=torch.long)
        done = torch.zeros(B * K, dtype=torch.bool)
        for st in range(maxlen):
            d = m.step(last, st, mem, mpad, exte, n_oov, cache, ew)
            lp = torch.log(d + 1e-9) / temp
            lp[:, PAD] = -1e9; lp[:, BOS] = -1e9; lp[:, UNK] = -1e9
            if st < minlen:
                lp[:, EOS] = -1e9
            v, ix = lp.topk(min(topk, lp.size(1)), -1)
            pr = F.softmax(v, -1)
            pick = ix.gather(1, torch.multinomial(pr, 1)).squeeze(1)
            pick = torch.where(done, torch.full_like(pick, EOS), pick)
            toks = torch.cat([toks, pick.unsqueeze(1)], 1)
            done = done | (pick == EOS)
            last = torch.where(pick < V, pick, torch.full_like(pick, UNK)).unsqueeze(1)
            if bool(done.all()):
                break
        tl = toks.tolist()
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
                cands.append((ws, 0.0))
            res[ch[b]] = cands
    return res


print("\n=== 2. sampled MBR (consensus decoding) ===", flush=True)
for nsamp, temp in [(12, 0.8), (12, 1.0), (20, 0.9)]:
    gs = sample_gen(m, TR, val, nsamp=nsamp, temp=temp)
    s = score({i: select(gs[i], 0.0, True) for i in val})
    L = np.mean([len(select(gs[i], 0.0, True)) for i in val])
    print(f"  nsamp{nsamp} temp{temp} MBR: G={s:.4f} len={L:.1f}  [{time.time()-t0:.0f}s]", flush=True)

print("\n=== 3. beam+sample union MBR ===", flush=True)
gb = generate(m, TR, val, beam=5, minlen=26, maxlen=54)
gs = sample_gen(m, TR, val, nsamp=12, temp=0.9)
uni = {i: gb[i] + gs[i] for i in val}
print(f"  union MBR: G={score({i: select(uni[i], 0.0, True) for i in val}):.4f}  [{time.time()-t0:.0f}s]", flush=True)
