"""Paired comparison of 1.5B G3 (rows.jsonl) against 3B arms (../math-g3-3b/rows.jsonl) on shared questions."""
import json, math
def p(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1, 2 * sum(math.comb(n, k) for k in range(min(f, b) + 1)) / 2 ** n)
S = {r["idx"]: r for r in map(json.loads, open("rows.jsonl"))}
B = {r["idx"]: r for r in map(json.loads, open("../math-g3-3b/rows.jsonl"))}
ids = [i for i in S if i in B]; n = len(ids)
def cmp(name, a, b):
    A = [a(i) for i in ids]; Bv = [b(i) for i in ids]
    f = sum(x and not y for x, y in zip(A, Bv)); br = sum(y and not x for x, y in zip(A, Bv))
    print(f"{name}: {100*sum(A)/n:.1f} vs {100*sum(Bv)/n:.1f}  diff {100*(sum(A)-sum(Bv))/n:+.1f}  (fixed {f} / broken {br}) p={p(f,br):.4g}")
print(f"n paired = {n}; 1.5B boxed rate {100*sum(S[i]['boxed'] for i in ids)/n:.1f}%")
cmp("H1/H2 1.5B G3 vs 3B greedy CoT", lambda i: S[i]["g3_ok"], lambda i: B[i]["cot_ok"])
cmp("H3 1.5B G3 vs 1.5B greedy CoT", lambda i: S[i]["g3_ok"], lambda i: S[i]["cot_ok"])
cmp("1.5B greedy vs 3B greedy", lambda i: S[i]["cot_ok"], lambda i: B[i]["cot_ok"])
cmp("3B G3 vs 1.5B G3", lambda i: B[i]["g3_ok"], lambda i: S[i]["g3_ok"])
c15 = sum(S[i]["cot_tok"] + S[i]["pal_tok"] for i in ids) * 1.54; c3 = sum(B[i]["cot_tok"] for i in ids) * 3.09
print(f"compute (tokens x params) 1.5B-G3 / 3B-greedy = {c15/c3:.2f}x")
