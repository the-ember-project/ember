import json, sys, math
from collections import Counter
sys.path.insert(0, "../m3-math"); import run as M
base = {r["idx"]: r for r in map(json.loads, open("../math-g3/rows.jsonl"))}
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl") if l.strip()]
def vote(c, others):  # same rule as math_g3.py: plurality, ties to greedy CoT, else first tied in order
    v = Counter(M.key(x) for x in [c] + others if x is not None)
    if not v: return c
    top = max(v.values()); best = [k for k, n in v.items() if n == top]
    if M.key(c) in best: return c
    if len(best) == 1: return next(x for x in others if M.key(x) == best[0])
    return next(x for x in others if M.key(x) in best)
def sign(f, b):
    n = f + b; k = min(f, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
A = {}
for r in rows:
    b = base[r["idx"]]; g = r["gold"]; c, p = b["cot"], b["pal"]
    arms = {"COT": c, "G3": vote(c, p), "A_SYM": vote(c, p[:2] + [r["sym"]]), "A_CS": vote(c, p[:2] + [r["cs"]]),
            "G3+SYM": vote(c, p + [r["sym"]]), "SYM": r["sym"], "CS": r["cs"], "PAL1": p[0]}
    for k, v in arms.items(): A.setdefault(k, []).append(M.same(v, g))
n = len(rows); print(f"n={n}")
print(f"SYM executes to a number: {100*sum(r['sym'] is not None for r in rows)/n:.1f}%  CS boxed: {100*sum(r['cs_boxed'] for r in rows)/n:.1f}%")
print(f"stored G3 reproduced: {sum(A['G3'][j] == base[r['idx']]['g3_ok'] for j, r in enumerate(rows))}/{n}")
for k, v in A.items(): print(f"{k:7s} {100*sum(v)/n:.1f}%")
def cmp(x, y, label):
    f = sum(a and not b for a, b in zip(A[x], A[y])); bk = sum(b and not a for a, b in zip(A[x], A[y]))
    print(f"{label}: {x}-{y} {100*(f-bk)/n:+.1f} (fixed {f} / broken {bk}) p={sign(f, bk):.4f}")
cmp("A_SYM", "G3", "H1"); cmp("G3", "A_CS", "H2"); cmp("G3+SYM", "G3", "expl"); cmp("A_SYM", "A_CS", "expl")
# exploratory: same-wrong-answer rate with greedy CoT, when both give a wrong number
for nm in ["PAL1", "SYM", "CS"]:
    both = [(base[r["idx"]]["cot"], (base[r["idx"]]["pal"][0] if nm == "PAL1" else r[nm.lower()]), r["gold"]) for r in rows]
    bw = [(x, y) for x, y, g in both if x is not None and y is not None and not M.same(x, g) and not M.same(y, g)]
    print(f"expl CoT vs {nm}: both wrong {len(bw)}, same wrong answer {sum(M.key(x)==M.key(y) for x,y in bw)}")
