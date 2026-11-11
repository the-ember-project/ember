import json, math, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl") if l.strip()]
n = len(rows)
def p(f, b):
    m = f + b; k = min(f, b)
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
c = [r["cot_ok"] for r in rows]; g = [r["g3_ok"] for r in rows]
f = sum(y and not x for x, y in zip(c, g)); b = sum(x and not y for x, y in zip(c, g))
pal1 = sum(r["pal_ok"][0] for r in rows)
print(f"n={n} greedy CoT {100*sum(c)/n:.1f}% | PAL sample1 {100*pal1/n:.1f}% | PAL any-of-3 {100*sum(any(r['pal_ok']) for r in rows)/n:.1f}%")
print(f"G3 {100*sum(g)/n:.1f}%  diff {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) sign p={p(f,b):.4g}")
print(f"validity: boxed rate {100*sum(r['boxed'] for r in rows)/n:.1f}%  (need >=90); CoT acc in 30-60")
ct = sum(r["cot_tok"] for r in rows); pt = sum(r["pal_tok"] for r in rows)
print(f"token cost G3/greedy = {(ct+pt)/ct:.2f}x; mean secs/q {sum(r['secs'] for r in rows)/n:.0f}")
by = defaultdict(lambda: [0, 0, 0])
for r in rows:
    by[r["level"]][0] += 1; by[r["level"]][1] += r["cot_ok"]; by[r["level"]][2] += r["g3_ok"]
for k in sorted(by): print(f"  {k}: n={by[k][0]} cot {by[k][1]} g3 {by[k][2]}")
