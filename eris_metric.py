import re

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
    if s is None:
        return []
    s = str(s).lower()
    s = _punc.sub("", s)
    return s.split()


def content(s):
    return [t for t in tok(s) if t not in STOP]


def lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        pv = 0
        for j, y in enumerate(b):
            if x == y:
                v = prev[j] + 1
            else:
                v = cur[j] if cur[j] >= prev[j + 1] else prev[j + 1]
            cur.append(v)
        prev = cur
    return prev[-1]


def rouge_l_f1(pt, rt):
    if not pt or not rt:
        return 0.0
    L = lcs_len(pt, rt)
    if L == 0:
        return 0.0
    p = L / len(pt)
    r = L / len(rt)
    return 2 * p * r / (p + r)


def cf(pred, ref):
    return rouge_l_f1(tok(pred), tok(ref))


def true_terms(trace, canon):
    return set(content(canon)) - set(content(trace))


def parse_terms(s):
    if s is None or (isinstance(s, float)):
        return set()
    out = set()
    for t in str(s).split(";"):
        t = t.strip().lower()
        if t and re.fullmatch(r"\w+", t):
            out.add(t)
    return out


def set_f1(P, T):
    if not P and not T:
        return 1.0
    if not P or not T:
        return 0.0
    i = len(P & T)
    if i == 0:
        return 0.0
    return 2 * i / (len(P) + len(T))


def tss(preds, refs, pred_sets, true_sets):
    G = sum(cf(p, r) for p, r in zip(preds, refs)) / len(refs)
    V = sum(set_f1(p, t) for p, t in zip(pred_sets, true_sets)) / len(true_sets)
    return max(0.02, min(1.0, 0.60 * G + 0.40 * V)), G, V
