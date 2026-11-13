"""1.5B G3 vs 3B greedy vs 1.5B greedy on GSM8K offsets 0-299 (paired, sign test)."""
import json, math, os
from collections import OrderedDict
def load(*fs):
    d = {}
    for f in fs:
        if os.path.exists(f):
            for l in open(f):
                if l.strip(): r = json.loads(l); d[r["idx"]] = r
    return d
B = "../gsm8k-baseline/"
g15 = load(B + "results_qwen1.5b_greedy_n100.jsonl", B + "results_qwen1.5b_greedy_off100_n200.jsonl")
g3b = load("../g3-scale/cot_3b.jsonl"); pk = load("pal_1.5b_k3.jsonl")
def eq(a, b): return a is not None and b is not None and abs(a - b) < 1e-4
def vote(c):
    s = OrderedDict()
    for v in c:
        if v is None: continue
        k = round(v, 4); s[k] = s.get(k, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def sign(b, c):
    k = min(b, c); return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)) if b + c else 1.0
ids = sorted(i for i in pk if i in g15 and i in g3b); n = len(ids)
pal_ok = sum(v is not None for i in ids for v in pk[i]["preds"][:3]) / max(1, 3 * n)
print(f"paired n={n}; PAL samples executing to a number: {100*pal_ok:.1f}%")
A = {"1.5B greedy": [eq(g15[i]["pred"], g15[i]["gold"]) for i in ids],
     "3B greedy": [eq(g3b[i]["pred"], g3b[i]["gold"]) for i in ids],
     "1.5B G3": [eq(vote([g15[i]["pred"]] + pk[i]["preds"][:3]), g15[i]["gold"]) for i in ids]}
for k, v in A.items(): print(f"{k:12s} {100*sum(v)/n:.1f}%")
def cmp(x, y, lab):
    f = sum(a and not b for a, b in zip(A[x], A[y])); b = sum(b and not a for a, b in zip(A[x], A[y]))
    print(f"{lab}: {x} - {y} = {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
cmp("1.5B G3", "3B greedy", "H1/H2"); cmp("1.5B G3", "1.5B greedy", "H3"); cmp("3B greedy", "1.5B greedy", "gap")
t15 = sum(g15[i]["tokens"] for i in ids) / n; t3 = sum(g3b[i]["tokens"] for i in ids) / n
tp = sum(pk[i]["tokens"] for i in ids) / n  # tokens field = all k samples
print(f"mean tokens: 1.5B greedy {t15:.0f}, 3B greedy {t3:.0f}, 1.5B PAL x3 {tp:.0f}")
print(f"compute (tokens x params) 1.5B-G3 / 3B-greedy = {1.5*(t15+tp)/(3*t3):.2f}x")
