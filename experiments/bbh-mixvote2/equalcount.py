"""EXPLORATORY (session #10, answering weekly review 2026-11-08): mix vs CoT-only at EQUAL sample count (5 samples).
Reference = maj5 sampled CoT. Sign test is paired mix vs maj5 (not vs greedy)."""
import sys
sys.argv = sys.argv[:1]
from analyze import R, T, vote, sign
mixes = {"maj5 CoT": (5, 0), "4C+1P": (4, 1), "3C+2P": (3, 2), "2C+3P": (2, 3), "5P": (0, 5)}
def run(name, rows):
    n = len(rows); ref = [vote(r["cot_samples"][:5]) == r["gold"] for r in rows]
    print(f"== {name} n={n} maj5CoT={sum(ref)/n:.3f}")
    for k, (c, p) in mixes.items():
        ok = [vote(r["cot_samples"][:c] + r["pal_samples"][:p]) == r["gold"] for r in rows]
        b = sum(x and not y for x, y in zip(ok, ref)); d = sum(y and not x for x, y in zip(ok, ref))
        print(f"   {k:9s} acc={sum(ok)/n:.3f} vs maj5 {100*(b-d)/n:+.1f} fixed={b} broken={d} p={sign(b,d):.3f}")
for t, rows in R.items(): run(t, rows)
run("POOLED", [r for t in T for r in R[t]])
