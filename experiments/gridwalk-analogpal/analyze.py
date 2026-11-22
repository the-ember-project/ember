"""Paired verdicts: analog-demo PAL (pal_analog.jsonl) vs gridwalk2 greedy CoT and original-demo PAL (same 200 tasks)."""
import json, math, os, re, sys
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, "..", "gridwalk2")
def load(p): return {r["idx"]: r for r in map(json.loads, open(p))}
def vote(ps):
    c = {}
    for p in ps:
        if p is not None: c[p] = c.get(p, 0) + 1
    return max(c, key=lambda k: c[k]) if c else None  # ties -> earliest
def signp(f, b):
    n = f + b; k = min(f, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
N = load(os.path.join(H, "pal_analog.jsonl")); O = load(os.path.join(G, "pal_s.jsonl")); C = load(os.path.join(G, "cot_greedy.jsonl"))
ids = sorted(set(N) & set(O) & set(C)); n = len(ids)
acc = {"cot_greedy": {i: C[i]["preds"][0] == C[i]["gold"] for i in ids},
       "vote5P_analog": {i: vote(N[i]["preds"]) == N[i]["gold"] for i in ids},
       "vote5P_orig": {i: vote(O[i]["preds"]) == O[i]["gold"] for i in ids},
       "pal1_analog": {i: N[i]["preds"][0] == N[i]["gold"] for i in ids},
       "pal1_orig": {i: O[i]["preds"][0] == O[i]["gold"] for i in ids}}
print("n", n, {k: round(100 * sum(v.values()) / n, 1) for k, v in acc.items()})
ns = [s for i in ids for s in N[i]["samples"]]
print("analog per-sample acc %.1f, exec rate %.1f, mean tokens %.0f" % (
    100 * sum(s["pred"] == N[i]["gold"] for i in ids for s in N[i]["samples"]) / len(ns),
    100 * sum(s["pred"] is not None for s in ns) / len(ns), sum(s["tokens"] for s in ns) / len(ns)))
print("analog samples using a direction list/modulo-4 simulator: %.1f%%" % (100 * sum(bool(re.search(r"%\s*4", s["text"])) for s in ns) / len(ns)))
def cmp(x, y, tag):
    f = sum(acc[x][i] and not acc[y][i] for i in ids); b = sum(acc[y][i] and not acc[x][i] for i in ids)
    print(f"{tag}: {x} - {y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={signp(f, b):.3g}")
cmp("vote5P_analog", "cot_greedy", "H1 (pass if >= +10, p<.05)")
cmp("vote5P_orig", "vote5P_analog", "H2 (pass if >= +5, p<.05)")
cmp("pal1_analog", "cot_greedy", "expl")
