# Verdicts for gsm8k-orm from per-question files (paired exact sign tests).
import json, math
from collections import Counter, defaultdict
from math import comb

def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def key(v): return None if v is None else round(v, 4)
def wvote(ans, w):  # weighted plurality, ties to earliest
    tot, first = defaultdict(float), {}
    for j, (x, wt) in enumerate(zip(ans, w)):
        k = key(x)
        if k is None or wt is None: continue
        tot[k] += wt; first.setdefault(k, j)
    if not tot: return None
    return max(tot, key=lambda k: (tot[k], -first[k]))
def sig(x): return 1 / (1 + math.exp(-x))
def sign_p(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

L = lambda p: {r["idx"]: r for r in map(json.loads, open(p))}
G = L("../gsm8k-g3confirm/cot_greedy_off900.jsonl"); S = L("cot_k7.jsonl"); B = L("cot3b_greedy.jsonl")
R = L("rm17_scores.jsonl"); P = L("../gsm8k-g3confirm/pal_k4_off900.jsonl")
items = [i for i in S if i in G and i in B and i in R]
print(f"n paired = {len(items)}")
acc = defaultdict(dict); toks = defaultdict(int)
nonfin = sum(x is None for i in items for x in R[i]["scores"])
for i in items:
    g = G[i]["gold"]; ans = [G[i]["pred"]] + S[i]["preds"]
    acc["greedy_1.5B"][i] = ok(G[i]["pred"], g)
    acc["greedy_3B"][i] = ok(B[i]["pred"], g)
    acc["SC8_1.5B"][i] = ok(wvote(ans, [1] * 8), g)
    acc["WSC8_1.5B"][i] = ok(wvote(ans, [None if x is None else sig(x) for x in R[i]["scores"]]), g)
    sc = R[i]["scores"]; best = max(range(8), key=lambda j: -1e9 if sc[j] is None else sc[j])
    acc["BoN8_1.5B"][i] = ok(ans[best], g)
    acc["oracle8"][i] = any(ok(x, g) for x in ans)
    if i in P: acc["G3_1.5B"][i] = ok(wvote([G[i]["pred"]] + P[i]["preds"][:3], [1] * 4), g)
    toks["1.5B"] += G[i]["tokens"] + S[i]["tokens"]; toks["3B"] += B[i]["tokens"]
for k, v in acc.items(): print(f"{k:12s} {100 * sum(v.values()) / len(v):5.1f}  (n={len(v)})")
ext = sum(x is not None for i in items for x in S[i]["preds"]) / (7 * len(items))
print(f"validity: 3B greedy {100 * sum(acc['greedy_3B'].values()) / len(items):.1f} (need 70-88); "
      f"sample answer extracted {100 * ext:.1f}% (need >=95); RM non-finite {nonfin} (need 0)")
print(f"decode tokens: 1.5B x8 {toks['1.5B']}, 3B greedy {toks['3B']}; "
      f"compute ratio (tokens x params, decode only) {toks['1.5B'] * 1.5 / (toks['3B'] * 3):.2f}x (+ RM 1.7B prefill of 8 solutions)")

def cmp(x, y, lab, thr=None, need_p=False):
    its = [i for i in items if i in acc[x] and i in acc[y]]
    d = 100 * (sum(acc[x][i] for i in its) - sum(acc[y][i] for i in its)) / len(its)
    f = sum(acc[x][i] and not acc[y][i] for i in its); b = sum(acc[y][i] and not acc[x][i] for i in its)
    p = sign_p(f, b); v = "" if thr is None else (" PASS" if d >= thr and (p < 0.05 or not need_p) else " FAIL")
    print(f"{lab}: {x} - {y} = {d:+.1f} ({f}/{b}, p={p:.3g}){v}")
cmp("WSC8_1.5B", "SC8_1.5B", "H1 (>= +2, p<.05)", 2, True)
cmp("WSC8_1.5B", "greedy_3B", "H2 (non-inf >= -2, point est.)", -2)
cmp("WSC8_1.5B", "greedy_3B", "H3 (>= +3, p<.05)", 3, True)
for x, y in [("SC8_1.5B", "greedy_1.5B"), ("WSC8_1.5B", "greedy_1.5B"), ("BoN8_1.5B", "SC8_1.5B"),
             ("greedy_3B", "greedy_1.5B"), ("SC8_1.5B", "greedy_3B"), ("G3_1.5B", "greedy_3B")]:
    cmp(x, y, "expl")
