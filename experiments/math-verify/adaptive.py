# Exploratory (session #36): ORM-guided adaptive stopping for the 1.5B (greedy first, then T=0.7 samples).
# Stop at k once the leading answer's ORM-weighted vote mass exceeds the runner-up by margin M (or k=8).
# Cost in "1.5B-sample units" (generated tokens / mean 1.5B greedy tokens); 3B counted at 2x per token.
import sys; sys.argv=[sys.argv[0]]
exec(open("analyze.py").read().split("n = len(items)")[0].replace('print(', '(lambda *a, **k: None)('))
from collections import defaultdict
g = G["1.5B"]; S = scores("1.5B", "rm17")["scores"]; g3 = G["3B"]
unit = sum(g[i]["toks"][0] for i in items) / len(items)
def lead(ans, w):
    m = defaultdict(float)
    for a, x in zip(ans, w):
        if a is not None: m[round(a, 4)] += x
    if not m: return None, 0.0
    o = sorted(m.items(), key=lambda t: -t[1]); return o[0][0], o[0][1] - (o[1][1] if len(o) > 1 else 0.0)
print(f"n={len(items)}  1.5B greedy tokens/q={unit:.0f}")
print(" M     acc   mean_k  cost(units)")
for M in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 99):
    c = k_tot = tok = 0
    for i in items:
        ans, s = g[i]["ans"], [sig(x) for x in S[str(i)]]
        for k in range(1, 9):
            a, mg = lead(ans[:k], s[:k])
            if mg >= M or k == 8: break
        c += ok(a, g[i]["gold"]); k_tot += k; tok += sum(g[i]["toks"][:k])
    print(f"{M:4}  {100*c/len(items):5.1f}  {k_tot/len(items):5.2f}  {tok/len(items)/unit:5.2f}")
t3 = sum(g3[i]["toks"][0] for i in items) / len(items)
print(f"3B greedy acc {100*sum(ok(g3[i]['ans'][0], g3[i]['gold']) for i in items)/len(items):.1f} cost {2*t3/unit:.2f} units")
