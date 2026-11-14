"""Paired comparison: 3B greedy (this job) vs m3-dpo3 BASE-1.5B and MIXR-1.5B on the same items."""
import json, math, os, sys
H = os.path.dirname(os.path.abspath(__file__))
def load(p):
    g, b = {}, {}
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if "idx" in r: g[r["idx"]] = r
        else: b[r["bbh_idx"]] = r
    return g, b
def sign(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
D = os.path.join(H, "../m3-dpo3/gpu-output/")
A = {"3B": load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "gpu-output/eval_BASE.jsonl")),
     "BASE": load(D + "eval_BASE.jsonl"), "MIXR": load(D + "eval_MIXR.jsonl")}
for t, part in [("GSM8K held-out", 0), ("BBH date", 1)]:
    ids = sorted(set.intersection(*[set(v[part]) for v in A.values()]))
    n = len(ids); acc = {k: [v[part][i]["correct"] for i in ids] for k, v in A.items()}
    print(f"== {t} n={n}: " + "  ".join(f"{k} {100*sum(x)/n:.1f}" for k, x in acc.items()))
    for x, y in [("MIXR", "3B"), ("3B", "MIXR"), ("3B", "BASE"), ("MIXR", "BASE")]:
        f = sum(a and not b for a, b in zip(acc[x], acc[y])); b = sum(b and not a for a, b in zip(acc[x], acc[y]))
        print(f"  {x}-{y}: {100*(sum(acc[x])-sum(acc[y]))/n:+.1f} (fixed {f} / broken {b}) p={sign(f, b):.4g}")
    if part == 0:
        L = {k: sum(v[0][i]["len"] for i in ids) / n for k, v in A.items()}
        gap = sum(acc["3B"]) - sum(acc["BASE"])
        print("  mean output chars:", {k: round(v) for k, v in L.items()},
              f" gap closed by MIXR: {(sum(acc['MIXR'])-sum(acc['BASE']))/gap:.2f}" if gap else "")
