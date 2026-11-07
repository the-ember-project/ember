"""G3 at other model sizes: vote(greedy CoT + 3 PAL samples), greedy wins ties. Usage: python analyze.py 0.5b|3b"""
import json, math, os, sys
from collections import OrderedDict
tag = sys.argv[1]
def load(f): return {r["idx"]: r for r in map(json.loads, open(f)) if r} if os.path.exists(f) else {}
g, pk = load(f"cot_{tag}.jsonl"), load(f"pal_{tag}.jsonl")
def eq(a, b): return a is not None and b is not None and abs(a - b) < 1e-4
def vote(c):
    s = OrderedDict()
    for v in c:
        if v is None: continue
        k = round(v, 4); s[k] = s.get(k, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def sign(b, c):
    k = min(b, c); return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)) if b + c else 1.0
ids = sorted(i for i in g if i in pk); n = len(ids)
print(f"model={tag} paired n={n}")
if not n: sys.exit()
K = len(next(iter(pk.values()))["preds"])
gt = sum(g[i]["tokens"] for i in ids) / n; pt = sum(pk[i]["tokens"] for i in ids) / n / K
S = {"greedy": (lambda i: g[i]["pred"], 1.0),
     "G3 vote(greedy+3PAL) [primary]": (lambda i: vote([g[i]["pred"]] + pk[i]["preds"][:3]), 1 + 3 * pt / gt),
     "[ref] vote(3PAL)": (lambda i: vote(pk[i]["preds"][:3]), 3 * pt / gt),
     "[ref] PAL sample 1": (lambda i: pk[i]["preds"][0], pt / gt)}
ok = {k: [eq(f(i), g[i]["gold"]) for i in ids] for k, (f, _) in S.items()}
for k, (_, c) in S.items():
    b = sum(x and not y for x, y in zip(ok[k], ok["greedy"])); d = sum(y and not x for x, y in zip(ok[k], ok["greedy"]))
    print(f"{k:32s} acc={sum(ok[k])/n:.3f} diff={100*(b-d)/n:+.1f} fixed={b} broken={d} p={sign(b,d):.4f} cost={c:.2f}x")
print(f"PAL unparsed rate={sum(x is None for i in ids for x in pk[i]['preds'][:3])/(3*n):.3f}")
