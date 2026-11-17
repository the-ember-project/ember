"""Verdicts for math-3b-sc (rules fixed in README before the run). Joins math-mix7 + math-sc7 (1.5B) with 3B rows."""
import json, math, os
H = os.path.dirname(os.path.abspath(__file__))
def rd(p): return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
B = rd(os.path.join(H, "..", "math-mix7", "rows.jsonl")); C = rd(os.path.join(H, "..", "math-sc7", "cs2.jsonl"))
G = rd(os.path.join(H, "g.jsonl")); S = rd(os.path.join(H, "s.jsonl"))
from collections import OrderedDict
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
P15, P3 = 1.54, 3.09
def report(ids, with_s):
    n = len(ids); A = {}
    for i in ids:
        r, c, g3 = B[i], C[i], G[i]; gold = r["gold"]
        arms = {"COT1.5": [r["cot"]], "SC7_1.5": [r["cot"]] + r["cs"] + c["cs2"], "MIX7_1.5": [r["cot"]] + r["cs"] + r["pal"],
                "G3_1.5": [r["cot"]] + r["pal"], "COT3": [g3["cot"]]}
        if with_s: arms["SC3_3"] = [g3["cot"]] + S[i]["cs"]
        for k, v in arms.items(): A.setdefault(k, []).append(ok(vote(v), gold))
    print(f"n={n}  3B greedy boxed {100*sum(G[i]['boxed'] for i in ids)/n:.1f}%" +
          (f"  3B sampled boxed {100*sum(sum(S[i]['boxed']) for i in ids)/(2*n):.1f}%" if with_s else ""))
    for k, v in A.items(): print(f"{k:9s} {100*sum(v)/n:.1f}%")
    def cmp(x, y, lab):
        f = sum(a and not b for a, b in zip(A[x], A[y])); b = sum(b and not a for a, b in zip(A[x], A[y]))
        print(f"{lab}: {x}-{y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
    cmp("SC7_1.5", "COT3", "H1 (non-inf, need >= -2)")
    if with_s:
        cmp("SC3_3", "SC7_1.5", "H2 (need >= +3, p<.05)"); cmp("SC7_1.5", "SC3_3", "H3 (non-inf, need >= -2)")
        cmp("SC3_3", "COT3", "expl")
    cmp("MIX7_1.5", "COT3", "expl"); cmp("G3_1.5", "COT3", "expl"); cmp("COT3", "COT1.5", "expl")
    t15 = sum(B[i]["cot_tok"] + sum(B[i]["cs_tok"]) + sum(C[i]["cs2_tok"]) for i in ids) * P15
    g = sum(G[i]["tok"] for i in ids) * P3
    print(f"compute (tokens x params) SC7_1.5 / COT3 = {t15/g:.2f}" +
          (f";  SC7_1.5 / SC3_3 = {t15/(g + sum(sum(S[i]['tok']) for i in ids)*P3):.2f}" if with_s else ""))
ids = [i for i in G if i in B and i in C]
both = [i for i in ids if i in S]
if len(both) == len(ids) == 250: report(ids, True)
else:
    print(f"partial: greedy {len(ids)}, sampled {len(both)} (verdicts need 250/250)")
    if ids: report(ids, False)
