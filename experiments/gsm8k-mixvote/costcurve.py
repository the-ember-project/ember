"""Exploratory: accuracy vs token cost for votes over a CoT samples + b PAL samples (offsets 300-599, post hoc)."""
import os, sys
sys.argv=[sys.argv[0]]
D=os.path.dirname(os.path.abspath(__file__)); exec(open(os.path.join(D,"analyze.py")).read().split("ids = sorted")[0])
ids = sorted(i for i in g if i in ck and i in pg and i in pk); n=len(ids); gold={i:g[i]["gold"] for i in ids}
G=[eq(g[i]["pred"],gold[i]) for i in ids]; gt=sum(g[i]["tokens"] for i in ids)/n
ct=sum(ck[i]["tokens"] for i in ids)/n/5; pt=sum(pk[i]["tokens"] for i in ids)/n/5
print(f"n={n} greedy={sum(G)/n:.3f}; per-sample tokens CoT={ct:.0f} PAL={pt:.0f} greedyCoT={gt:.0f}")
print("a_CoT b_PAL  acc   diff  fixed broken p      cost(x greedy)")
for a in range(0,6):
  for b in range(0,6):
    if a+b==0: continue
    ok=[eq(vote([(v,1) for v in ck[i]["preds"][:a]]+[(v,1) for v in pk[i]["preds"][:b]]),gold[i]) for i in ids]
    f=sum(o and not x for o,x in zip(ok,G)); br=sum(x and not o for o,x in zip(ok,G))
    print(f"{a} {b}  {sum(ok)/n:.3f} {100*(sum(ok)-sum(G))/n:+5.1f} {f:4d} {br:4d} {sign(f,br):.3f} {(a*ct+b*pt)/gt:5.2f}")
