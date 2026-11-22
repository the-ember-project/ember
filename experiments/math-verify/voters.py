# Exploratory (session #38, answering review 2026-11-22): does voter count matter at 2048 tokens? Paired sign tests
# of SC1 (greedy) / SC4 / SC8 from the stored math-verify samples (greedy first, then T=0.7 samples), + truncation rate.
import sys; sys.argv=[sys.argv[0]]
exec(open("analyze.py").read().split("n = len(items)")[0].replace('print(', '(lambda *a, **k: None)('))
import math
def sp(f, b):
    n = f + b; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
for size in ("1.5B", "3B"):
    g = G[size]
    C = {k: [ok(wvote(g[i]["ans"][:k], [1]*k), g[i]["gold"]) for i in items] for k in (1, 4, 8)}
    tr = [t >= 2048 for i in items for t in g[i]["toks"]]
    print(f"{size}: per-sample truncated (>=2048 tok) {100*sum(tr)/len(tr):.1f}%; greedy truncated "
          f"{100*sum(g[i]['toks'][0] >= 2048 for i in items)/len(items):.1f}%; acc SC1/SC4/SC8 "
          + "/".join(f"{100*sum(C[k])/len(items):.1f}" for k in (1, 4, 8)))
    for a, b in ((8, 1), (4, 1), (8, 4)):
        f = sum(x and not y for x, y in zip(C[a], C[b])); bb = sum(y and not x for x, y in zip(C[a], C[b]))
        print(f"   SC{a}-SC{b}: {100*(sum(C[a])-sum(C[b]))/len(items):+.1f} ({f}/{bb}, p={sp(f, bb):.3g})")
