"""Verdicts for math-mix7 (rules fixed in README before the run)."""
import json, math, sys
from collections import OrderedDict
R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl") if l.strip()]
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def vote(c):  # plurality, ties go to the earliest voter (greedy first)
    s = OrderedDict()
    for v in c:
        k = key(v)
        if k is None: continue
        s[k] = s.get(k, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def sign(f, b):
    n = f + b; k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
n = len(R); A = {}
for r in R:
    c, p, s = r["cot"], r["pal"], r["cs"]
    for name, vs in {"COT": [c], "G3": [c] + p, "SC4": [c] + s, "MIX7": [c] + s + p}.items():
        A.setdefault(name, []).append(ok(vote(vs), r["gold"]))
print(f"n={n}  greedy boxed {100*sum(r['boxed'] for r in R)/n:.1f}%  "
      f"PAL exec {100*sum(x is not None for r in R for x in r['pal'])/(3*n):.1f}%")
for k, v in A.items(): print(f"{k:5s} {100*sum(v)/n:.1f}%")
def cmp(x, y, lab):
    f = sum(a and not b for a, b in zip(A[x], A[y])); b = sum(b and not a for a, b in zip(A[x], A[y]))
    print(f"{lab}: {x}-{y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
cmp("MIX7", "G3", "H1"); cmp("G3", "COT", "H2"); cmp("MIX7", "SC4", "H3"); cmp("G3", "SC4", "expl"); cmp("SC4", "COT", "expl"); cmp("MIX7", "COT", "expl")
ct = sum(r["cot_tok"] for r in R) / n; pt = sum(sum(r["pal_tok"]) for r in R) / n; st = sum(sum(r["cs_tok"]) for r in R) / n
print(f"cost vs greedy tokens: G3 {(ct+pt)/ct:.2f}x  SC4 {(ct+st)/ct:.2f}x  MIX7 {(ct+pt+st)/ct:.2f}x")
