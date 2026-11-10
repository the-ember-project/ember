"""EXPLORATORY (session #14): label-free format routing on saved samples (5 CoT + 5 PAL per question, T=0.7).
Tasks: GSM8K 300-599 (gsm8k-mixvote), BBH tracking_3 / logical_deduction_5 / date (bbh-mixvote2), gridwalk2.
Routers (no gold labels used for routing):
  TASK-k : on the first k questions of a task, mean within-format agreement a_F = (#samples equal to F's plurality)/5
           (abstentions count as disagreement). If a_P - a_C > d -> vote(5P); a_C - a_P > d -> vote(5C); else vote(3C+2P...).
           Evaluated on the remaining questions only.
  PERQ   : per question, take the answer of the format with the larger plurality count (tie -> MIX10 vote).
Compared on the same evaluation questions against MIX10, equal-count mixes, vote(5C), vote(5P), greedy CoT."""
import json, os
from collections import Counter
from math import comb
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def norm(x):
    if x is None: return None
    if isinstance(x, float): return f"{x:.4f}"
    return str(x)
def jl(p): return [json.loads(l) for l in open(os.path.join(E, p))]
def tasks():
    T = {}
    c = {r["idx"]: r for r in jl("gsm8k-mixvote/cot_k5_off300_n300.jsonl")}
    p = {r["idx"]: r for r in jl("gsm8k-mixvote/pal_k5_off300_n300.jsonl")}
    g = {r["idx"]: r for r in jl("gsm8k-mixvote/cot_greedy_off300_n300.jsonl")}
    ids = [i for i in c if i in p and i in g]  # file order
    T["gsm8k"] = [dict(gold=norm(c[i]["gold"]), C=[norm(x) for x in c[i]["preds"][:5]], P=[norm(x) for x in p[i]["preds"][:5]],
                       G=norm(g[i]["preds"][0])) for i in ids]
    for t in ["tracking_shuffled_objects_three_objects", "logical_deduction_five_objects", "date_understanding"]:
        T[t[:12]] = [dict(gold=r["gold"], C=r["cot_samples"][:5], P=r["pal_samples"][:5], G=r["cot_greedy"])
                     for r in jl(f"bbh-mixvote2/gpu-output/{t}.jsonl")]
    c = {r["idx"]: r for r in jl("gridwalk2/cot_s.jsonl")}; p = {r["idx"]: r for r in jl("gridwalk2/pal_s.jsonl")}
    g = {r["idx"]: r for r in jl("gridwalk2/cot_greedy.jsonl")}
    T["gridwalk2"] = [dict(gold=c[i]["gold"], C=c[i]["preds"][:5], P=p[i]["preds"][:5], G=g[i]["preds"][0]) for i in sorted(c)]
    return T
def vote(v):
    v = [x for x in v if x is not None]
    if not v: return None
    cnt = Counter(v); b = max(cnt.values()); return next(x for x in v if cnt[x] == b)
def top(v):
    v = [x for x in v if x is not None]; return max(Counter(v).values()) if v else 0
def signp(a, b):
    n = a + b; k = min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def perq(r):
    tc, tp = top(r["C"]), top(r["P"])
    return vote(r["P"]) if tp > tc else vote(r["C"]) if tc > tp else vote(r["C"] + r["P"])
K, D = 30, 0.10
pooled = {}
for name, rows in tasks().items():
    cal, ev = rows[:K], rows[K:]
    aC = sum(top(r["C"]) for r in cal) / (5 * K); aP = sum(top(r["P"]) for r in cal) / (5 * K)
    choice = "5P" if aP - aC > D else "5C" if aC - aP > D else "3C+2P"
    arms = {"greedy": lambda r: r["G"], "vote5C": lambda r: vote(r["C"]), "vote5P": lambda r: vote(r["P"]),
            "3C+2P": lambda r: vote(r["C"][:3] + r["P"][:2]), "2C+3P": lambda r: vote(r["C"][:2] + r["P"][:3]),
            "MIX10": lambda r: vote(r["C"] + r["P"]), "PERQ(10)": perq,
            "TASK-router(5)": (lambda r: vote(r["P"])) if choice == "5P" else (lambda r: vote(r["C"])) if choice == "5C"
                              else (lambda r: vote(r["C"][:3] + r["P"][:2]))}
    ok = {a: [f(r) == r["gold"] for r in ev] for a, f in arms.items()}
    for a in ok: pooled.setdefault(a, []).extend(ok[a])
    print(f"== {name}: cal agreement C={aC:.2f} P={aP:.2f} -> {choice}; eval n={len(ev)}")
    print("   " + "  ".join(f"{a}={100*sum(v)/len(v):.1f}" for a, v in ok.items()))
def cmp(a, b, d):
    fx = sum(x and not y for x, y in zip(d[a], d[b])); br = sum(y and not x for x, y in zip(d[a], d[b]))
    print(f"   {a} - {b}: {100*(fx-br)/len(d[a]):+.1f} ({fx}/{br}) p={signp(fx, br):.4f}")
n = len(pooled["greedy"])
print(f"== POOLED eval n={n} (tasks unequal size)")
print("   " + "  ".join(f"{a}={100*sum(v)/n:.1f}" for a, v in pooled.items()))
for a, b in [("TASK-router(5)", "3C+2P"), ("TASK-router(5)", "2C+3P"), ("TASK-router(5)", "MIX10"), ("PERQ(10)", "MIX10"),
             ("TASK-router(5)", "greedy"), ("MIX10", "greedy")]:
    cmp(a, b, pooled)
