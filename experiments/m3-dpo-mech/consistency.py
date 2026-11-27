"""Exploratory: are fixes/breaks consistent across the 3 DPO seeds? Compare to binomial independence."""
import json, collections, math
from analyze import rd, seeds, stratum, g3
B = {s: rd(b) for s, (b, m) in seeds.items()}; M = {s: rd(m) for s, (b, m) in seeds.items()}
ids = [i for i in g3 if len({B[s][i]['correct'] for s in B}) == 1]   # BASE agrees across runs
out = []
for st in ['A', 'B', 'C', 'AG', 'WG']:
    its = [i for i in ids if stratum(i, B['R1'][i]['correct']) == st]
    k = collections.Counter(sum(M[s][i]['correct'] != B[s][i]['correct'] for s in B) for i in its)
    n = len(its); p = sum(j * k[j] for j in k) / (3 * n)
    exp = [n * math.comb(3, j) * p**j * (1-p)**(3-j) for j in range(4)]
    out.append(f"{st}: n={n} changed-by-j-seeds {[k[j] for j in range(4)]} vs indep {[round(e) for e in exp]}")
# seed-to-seed churn between trained models
for a, b in [('R1', 'R2'), ('R1', 'R3'), ('R2', 'R3')]:
    d = sum(M[a][i]['correct'] != M[b][i]['correct'] for i in g3)
    out.append(f"MIXR {a} vs {b}: disagree on correctness {d}/1319")
d = sum(B['R1'][i]['correct'] != B['R2'][i]['correct'] for i in g3); out.append(f"BASE R1 vs R2 run nondeterminism: {d}/1319")
print('\n'.join(out)); open('consistency.txt', 'w').write('\n'.join(out) + '\n')
