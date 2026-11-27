"""Exploratory: plurality vote over the 3 MIXR seeds' greedy answers (tie -> R1) vs single seeds."""
import collections
from analyze import rd, seeds, g3, eq
M = {s: rd(m) for s, (b, m) in seeds.items()}; B = rd(seeds['R1'][0])
acc = {s: sum(M[s][i]['correct'] for i in g3) / 13.19 for s in M}
ens = 0
for i in g3:
    preds = [M[s][i]['pred'] for s in ('R1', 'R2', 'R3')]
    c = collections.Counter(round(float(p), 4) for p in preds if p is not None)
    top = c.most_common(1)[0][0] if c else None
    if c and c.most_common(1)[0][1] == 1: top = preds[0]
    ens += eq(top, g3[i]['gold'])
out = f"BASE {sum(B[i]['correct'] for i in g3)/13.19:.2f}  seeds {', '.join(f'{k} {v:.2f}' for k,v in acc.items())}  ENS3 {ens/13.19:.2f}  BASE G3 {sum(g3[i]['correct'] for i in g3)/13.19:.2f}"
print(out); open('ensemble.txt', 'w').write(out + '\n')
