"""EXPLORATORY post-hoc: abstention-aware cheap mixes on bbh-mixvote2 data (rules chosen after seeing H3 fail).
Cost is in units of greedy-CoT tokens, using per-task mean tokens of each sample type."""
import json, os, sys
sys.argv = sys.argv[:1]
from analyze import R, T, vote, sign
def cost(r, nc_samp, np_samp, greedy_cot=1):
    return (greedy_cot * r["tok"]["cg"] + nc_samp * r["tok"]["cs"] / 5 + np_samp * r["tok"]["ps"] / 5) / r["tok"]["cg"]
def rules():
    def C(r):   # original cheap mix (H3): 1 sampled CoT + 3 PAL
        return vote(r["cot_samples"][:1] + r["pal_samples"][:3]), cost(r, 1, 3, 0)
    def G3(r):  # greedy CoT first (wins ties) + 3 PAL
        return vote([r["cot_greedy"]] + r["pal_samples"][:3]), cost(r, 0, 3)
    def ABST(r):  # if <2 of 3 PAL parse, answer = greedy CoT; else vote(greedy CoT + 3 PAL)
        p = r["pal_samples"][:3]
        if sum(v is not None for v in p) < 2: return r["cot_greedy"], cost(r, 0, 3)
        return vote([r["cot_greedy"]] + p), cost(r, 0, 3)
    def ADAPT(r):  # 3 PAL; if <2 parse, buy 4 CoT samples instead and vote greedy+4 CoT (+ any PAL)
        p = r["pal_samples"][:3]
        if sum(v is not None for v in p) < 2: return vote([r["cot_greedy"]] + r["cot_samples"][:4] + p), cost(r, 4, 3)
        return vote([r["cot_greedy"]] + p), cost(r, 0, 3)
    def S2(r): return vote(r["cot_samples"] + r["pal_samples"]), cost(r, 5, 5, 0)
    return dict(C=C, G3=G3, ABST=ABST, ADAPT=ADAPT, S2=S2)
def report(name, rows):
    n = len(rows); G = [r["cot_greedy"] == r["gold"] for r in rows]
    print(f"== {name} n={n} greedy={sum(G)/n:.3f}")
    for k, f in rules().items():
        out = [f(r) for r in rows]; ok = [a == r["gold"] for (a, _), r in zip(out, rows)]
        b = sum(x and not g for x, g in zip(ok, G)); c = sum(g and not x for x, g in zip(ok, G))
        print(f"   {k:6s} acc={sum(ok)/n:.3f} diff={100*(sum(ok)-sum(G))/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f} cost={sum(x for _, x in out)/n:.2f}x")
for t, rows in R.items(): report(t, rows)
report("POOLED", [r for t in T for r in R[t]])
