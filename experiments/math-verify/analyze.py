"""Verdicts for math-verify (rules fixed in README before the run). Usage: python analyze.py [outdir]"""
import gzip, json, math, os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "gpu-output", "out")

def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def wvote(ans, w):  # weighted plurality, ties to the earliest
    s, first = defaultdict(float), {}
    for j, (v, x) in enumerate(zip(ans, w)):
        if v is None: continue
        k = round(v, 4); s[k] += x; first.setdefault(k, j)
    return max(s, key=lambda k: (s[k], -first[k])) if s else None
def sig(x): return 1 / (1 + math.exp(-x))
def sign_p(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
def load(size):
    p = os.path.join(D, f"gen_{size}.jsonl.gz")
    return {r["idx"]: r for r in (json.loads(l) for l in gzip.open(p, "rt"))} if os.path.exists(p) else None
def scores(size, rm):
    p = os.path.join(D, f"rm_{size}_{rm}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

G = {s: load(s) for s in ("1.5B", "3B")}
if G["1.5B"] is None: sys.exit(f"no gen_1.5B in {D}")
items = sorted(G["1.5B"])
acc = {}  # name -> {idx: bool}
for size in ("1.5B", "3B"):
    g = G[size]
    if g is None: continue
    acc[f"greedy_{size}"] = {i: ok(g[i]["ans"][0], g[i]["gold"]) for i in items}
    acc[f"SC8_{size}"] = {i: ok(wvote(g[i]["ans"], [1] * len(g[i]["ans"])), g[i]["gold"]) for i in items}
    acc[f"pass8_{size}"] = {i: any(ok(v, g[i]["gold"]) for v in g[i]["ans"]) for i in items}
    print(f"{size}: greedy unboxed {100*sum(not g[i]['boxed'][0] for i in items)/len(items):.1f}%  "
          f"mean tokens {sum(sum(g[i]['toks']) for i in items)/len(items)/len(g[items[0]]['toks']):.0f}")
    for rm in ("rm17", "rm06"):
        sc = scores(size, rm)
        if sc is None: continue
        S = sc["scores"]
        acc[f"WSC8_{size}_{rm}"] = {i: ok(wvote(g[i]["ans"], [sig(x) for x in S[str(i)]]), g[i]["gold"]) for i in items}
        acc[f"BoN8_{size}_{rm}"] = {i: ok(g[i]["ans"][max(range(len(S[str(i)])), key=lambda j: S[str(i)][j])], g[i]["gold"]) for i in items}
        pos = [x for i in items for v, x in zip(g[i]["ans"], S[str(i)]) if ok(v, g[i]["gold"])]
        neg = [x for i in items for v, x in zip(g[i]["ans"], S[str(i)]) if not ok(v, g[i]["gold"])]
        print(f"{size} {rm}: AUROC {auroc(pos, neg):.3f}  nonfinite fp16 {sc['n_nonfinite_fp16']} final {sc['n_nonfinite_final']}")
n = len(items)
print(f"n={n}")
for k, v in acc.items(): print(f"  {k:18s} {100*sum(v.values())/n:5.1f}")
def cmp(x, y, lab, thr=None, need_p=False):
    if x not in acc or y not in acc: print(lab, "(missing)"); return
    f = sum(acc[x][i] and not acc[y][i] for i in items); b = sum(acc[y][i] and not acc[x][i] for i in items)
    d = 100 * (f - b) / n; p = sign_p(f, b)
    verdict = "" if thr is None else (" PASS" if d >= thr and (p < 0.05 or not need_p) else " FAIL")
    print(f"{lab}: {x} - {y} = {d:+.1f} (fixed {f} / broken {b}) p={p:.4g}{verdict}")
cmp("WSC8_1.5B_rm17", "SC8_1.5B", "H1 (>= +2, p<.05)", 2, True)
cmp("WSC8_1.5B_rm17", "greedy_3B", "H2 (non-inf >= -2)", -2)
cmp("WSC8_1.5B_rm17", "SC8_3B", "H3 (non-inf >= -2)", -2)
for x, y in [("BoN8_1.5B_rm17", "SC8_1.5B"), ("SC8_1.5B", "greedy_1.5B"), ("SC8_3B", "greedy_3B"),
             ("WSC8_3B_rm17", "SC8_3B"), ("WSC8_1.5B_rm06", "SC8_1.5B"), ("greedy_3B", "greedy_1.5B")]:
    cmp(x, y, "expl")
lv = defaultdict(list)
for i in items: lv[G["1.5B"][i]["level"]].append(i)
for L in sorted(lv):
    print(L, len(lv[L]), " ".join(f"{k}={100*sum(acc[k][i] for i in lv[L])/len(lv[L]):.0f}" for k in acc if not k.startswith("pass")))
