"""Paired analysis for gridwalk. Files: cot_greedy.jsonl (k=1), cot_s.jsonl (k=5, T=0.7), pal_s.jsonl (k=5, T=0.7).
Plurality vote over parsed (x,y) strings; None = abstain; ties -> earliest-listed sample wins. Prints acc, diff vs
greedy, fixed/broken, exact two-sided sign-test p, token cost relative to greedy."""
import json, sys, os
from collections import Counter
from math import comb

D = sys.argv[1] if len(sys.argv) > 1 else "."
def load(f):
    p = os.path.join(D, f)
    return {r["idx"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
G, C, P = load("cot_greedy.jsonl"), load("cot_s.jsonl"), load("pal_s.jsonl")
ids = sorted(set(G) & set(C) & set(P))

def vote(preds):
    v = [p for p in preds if p is not None]
    if not v: return None
    cnt = Counter(v); best = max(cnt.values())
    return next(p for p in v if cnt[p] == best)

def signp(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def tok(r, m): return sum(s["tokens"] for s in r["samples"][:m])
arms = {
    "greedy": lambda i: ([G[i]["preds"][0]], tok(G[i], 1)),
    "MIX10 vote(5C+5P) [H1]": lambda i: (C[i]["preds"][:5] + P[i]["preds"][:5], tok(C[i], 5) + tok(P[i], 5)),
    "G3 vote(greedy+3P) [H2]": lambda i: ([G[i]["preds"][0]] + P[i]["preds"][:3], tok(G[i], 1) + tok(P[i], 3)),
    "vote(3C+2P) [H3 arm]": lambda i: (C[i]["preds"][:3] + P[i]["preds"][:2], tok(C[i], 3) + tok(P[i], 2)),
    "vote(5C) [H3 ref]": lambda i: (C[i]["preds"][:5], tok(C[i], 5)),
    "vote(2C+3P)": lambda i: (C[i]["preds"][:2] + P[i]["preds"][:3], tok(C[i], 2) + tok(P[i], 3)),
    "vote(5P)": lambda i: (P[i]["preds"][:5], tok(P[i], 5)),
    "PAL sample1": lambda i: (P[i]["preds"][:1], tok(P[i], 1)),
    "CoT sample1": lambda i: (C[i]["preds"][:1], tok(C[i], 1)),
}
res = {}
for name, f in arms.items():
    ok, t = {}, 0
    for i in ids:
        preds, tk = f(i); ok[i] = vote(preds) == G[i]["gold"]; t += tk
    res[name] = (ok, t)
base, bt = res["greedy"]
print(f"gridwalk paired n={len(ids)}")
def cmp(a, b):
    A, B = res[a][0], res[b][0]
    fx = sum(A[i] and not B[i] for i in ids); br = sum(B[i] and not A[i] for i in ids)
    return 100 * (fx - br) / max(len(ids), 1), fx, br, signp(fx, br)
for name in arms:
    ok, t = res[name]; d, fx, br, p = cmp(name, "greedy")
    print(f"{name:28s} acc={sum(ok.values())/max(len(ids),1):.3f} diff={d:+.1f} fixed={fx} broken={br} p={p:.4f} cost={t/max(bt,1):.2f}x")
d, fx, br, p = cmp("vote(3C+2P) [H3 arm]", "vote(5C) [H3 ref]")
print(f"H3 equal-count: vote(3C+2P) - vote(5C) = {d:+.1f} ({fx}/{br}) p={p:.4f}")
pas = [any(x == G[i]["gold"] for x in C[i]["preds"] + P[i]["preds"] + G[i]["preds"]) for i in ids]
print(f"pass@11 (any sample correct) = {sum(pas)/max(len(ids),1):.3f}; PAL abstain rate = "
      f"{sum(x is None for i in ids for x in P[i]['preds'])/max(5*len(ids),1):.3f}")
