"""Verdicts for math-sc7 (rules fixed in README before the run). Joins math-mix7 rows with cs2.jsonl."""
import json, math, os, sys
from collections import OrderedDict
H = os.path.dirname(os.path.abspath(__file__))
B = {r["idx"]: r for r in (json.loads(l) for l in open(os.path.join(H, "..", "math-mix7", "rows.jsonl")) if l.strip())}
C = [json.loads(l) for l in open(os.path.join(H, "cs2.jsonl")) if l.strip()]
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def vote(c):
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
A = {}; n = len(C); boxed = 0
for c in C:
    r = B[c["idx"]]; g = r["gold"]; boxed += sum(c["cs2_boxed"])
    arms = {"COT": [r["cot"]], "SC4": [r["cot"]] + r["cs"], "SC4b": [r["cot"]] + c["cs2"],
            "G3": [r["cot"]] + r["pal"], "MIX7": [r["cot"]] + r["cs"] + r["pal"], "SC7": [r["cot"]] + r["cs"] + c["cs2"]}
    for k, v in arms.items(): A.setdefault(k, []).append(ok(vote(v), g))
print(f"n={n}  new sampled-CoT boxed rate {100*boxed/(3*n):.1f}%")
for k, v in A.items(): print(f"{k:5s} {100*sum(v)/n:.1f}%")
def cmp(x, y, lab):
    f = sum(a and not b for a, b in zip(A[x], A[y])); b = sum(b and not a for a, b in zip(A[x], A[y]))
    print(f"{lab}: {x}-{y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
cmp("MIX7", "SC7", "H1"); cmp("SC7", "SC4", "H2"); cmp("SC4b", "SC4", "expl(seed check)"); cmp("SC7", "COT", "expl")
ct = sum(B[c["idx"]]["cot_tok"] for c in C) / n; pt = sum(sum(B[c["idx"]]["pal_tok"]) for c in C) / n
s1 = sum(sum(B[c["idx"]]["cs_tok"]) for c in C) / n; s2 = sum(sum(c["cs2_tok"]) for c in C) / n
print(f"cost vs greedy tokens: MIX7 {(ct+s1+pt)/ct:.2f}x  SC7 {(ct+s1+s2)/ct:.2f}x")
