"""Analysis for bbh-mixvote (rules fixed in README). python analyze.py [dir]"""
import json, math, os, sys
from collections import OrderedDict
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-output")
T = ["tracking_shuffled_objects_three_objects", "logical_deduction_five_objects", "date_understanding"]
R = {t: [json.loads(l) for l in open(os.path.join(D, t + ".jsonl"))] for t in T if os.path.exists(os.path.join(D, t + ".jsonl"))}
def vote(c):
    s = OrderedDict()
    for v in c:
        if v is not None: s[v] = s.get(v, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def sign(b, c):
    k = min(b, c); return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)) if b + c else 1.0
A = {"greedy CoT": lambda r: r["cot_greedy"], "C vote(1CoT+3PAL) [H3]": lambda r: vote(r["cot_samples"][:1] + r["pal_samples"][:3]),
     "S2 vote(5CoT+5PAL)": lambda r: vote(r["cot_samples"] + r["pal_samples"]),
     "S3 maj5 CoT": lambda r: vote(r["cot_samples"]), "greedy PAL": lambda r: r["pal_greedy"],
     "[expl] maj5 PAL": lambda r: vote(r["pal_samples"]),
     "[expl] vote(2CoT+5PAL)": lambda r: vote(r["cot_samples"][:2] + r["pal_samples"]),
     "[expl] vote(greedyCoT+greedyPAL+5CoT)": lambda r: vote([r["cot_greedy"], r["pal_greedy"]] + r["cot_samples"])}
def report(name, rows):
    n = len(rows)
    ok = {k: [f(r) == r["gold"] for r in rows] for k, f in A.items()}; G = ok["greedy CoT"]
    tg = sum(r["tok"]["cg"] for r in rows) / n
    print(f"== {name} n={n}  tokens/q: greedyCoT {tg:.0f}, 5CoT {sum(r['tok']['cs'] for r in rows)/n:.0f}, 5PAL {sum(r['tok']['ps'] for r in rows)/n:.0f}, greedyPAL {sum(r['tok']['pg'] for r in rows)/n:.0f}")
    print(f"   unparsed: CoT greedy {sum(r['cot_greedy'] is None for r in rows)}, PAL greedy {sum(r['pal_greedy'] is None for r in rows)}, PAL samples {sum(v is None for r in rows for v in r['pal_samples'])}/{5*n}")
    s2 = [vote(r["cot_samples"] + r["pal_samples"]) == r["gold"] for r in rows]; s3 = [vote(r["cot_samples"]) == r["gold"] for r in rows]
    b = sum(x and not y for x, y in zip(s2, s3)); c = sum(y and not x for x, y in zip(s2, s3))
    print(f"   [H2] S2 vs S3: diff={100*(b-c)/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f}")
    tc = sum(r["tok"]["cs"] / 5 + 3 * r["tok"]["ps"] / 5 for r in rows) / max(1, sum(r["tok"]["cg"] for r in rows))
    print(f"   [H3] cost of C = {tc:.2f}x greedy; unparsed greedy CoT = {sum(r['cot_greedy'] is None for r in rows)/n:.1%}")
    for k in A:
        b = sum(x and not g for x, g in zip(ok[k], G)); c = sum(g and not x for x, g in zip(ok[k], G))
        print(f"   {k:40s} acc={sum(ok[k])/n:.3f} diff={100*(sum(ok[k])-sum(G))/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f}")
    print(f"   [expl] pass@10 = {sum(r['gold'] in r['cot_samples'] + r['pal_samples'] for r in rows)/n:.3f}")
for t, rows in R.items(): report(t, rows)
if len(R) > 1: report("POOLED", [r for t in T if t in R for r in R[t]])
