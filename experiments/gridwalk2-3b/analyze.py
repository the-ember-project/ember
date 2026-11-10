"""gridwalk2-3b paired analysis: greedy CoT (G), greedy PAL (PG), 3 PAL samples T=0.7 (P). Plurality vote, ties ->
earliest listed (G3: greedy CoT first). Exact two-sided sign test vs greedy CoT."""
import json, os
from collections import Counter
from math import comb
def load(f): return {r["idx"]: r for r in map(json.loads, open(f))} if os.path.exists(f) else {}
G, PG, P = load("cot_greedy.jsonl"), load("pal_greedy.jsonl"), load("pal_s.jsonl")
ids = sorted(set(G) & set(PG) & set(P))
def vote(v):
    v = [x for x in v if x is not None]
    if not v: return None
    c = Counter(v); b = max(c.values()); return next(x for x in v if c[x] == b)
def signp(a, b):
    n = a + b; k = min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
tok = lambda r, m: sum(s["tokens"] for s in r["samples"][:m])
arms = {"greedy CoT": lambda i: ([G[i]["preds"][0]], tok(G[i], 1)),
        "greedy PAL [H2]": lambda i: ([PG[i]["preds"][0]], tok(PG[i], 1)),
        "G3 vote(greedyCoT+3P) [H1]": lambda i: ([G[i]["preds"][0]] + P[i]["preds"][:3], tok(G[i], 1) + tok(P[i], 3)),
        "vote(3P)": lambda i: (P[i]["preds"][:3], tok(P[i], 3)),
        "vote(greedyPAL+3P)": lambda i: ([PG[i]["preds"][0]] + P[i]["preds"][:3], tok(PG[i], 1) + tok(P[i], 3))}
n = len(ids); print(f"gridwalk2-3b paired n={n}")
if n:
    gold = {i: G[i]["gold"] for i in ids}; base = {i: G[i]["preds"][0] == gold[i] for i in ids}
    bt = sum(tok(G[i], 1) for i in ids)
    for a, f in arms.items():
        ok = {}; t = 0
        for i in ids:
            pr, tk = f(i); ok[i] = vote(pr) == gold[i]; t += tk
        fx = sum(ok[i] and not base[i] for i in ids); br = sum(base[i] and not ok[i] for i in ids)
        print(f"{a:28s} acc={sum(ok.values())/n:.3f} diff={100*(fx-br)/n:+.1f} fixed={fx} broken={br} p={signp(fx,br):.4f} cost={t/bt:.2f}x")
    skip = sum(G[i]["samples"][0]["tokens"] < 30 for i in ids) / n
    ab = sum(x is None for i in ids for x in P[i]["preds"][:3]) / (3 * n)
    print(f"greedy CoT skip rate (<30 tokens) = {skip:.3f} (pre-registered: >0.20 flags degenerate baseline); PAL sample abstain = {ab:.3f}")
