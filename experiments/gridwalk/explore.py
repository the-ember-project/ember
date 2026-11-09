"""Exploratory (session #12): greedy skips reasoning on ~half of questions. Breakdown + extra comparisons."""
import json
from collections import Counter
from math import comb
L=lambda f:{r['idx']:r for r in map(json.loads,open(f))}
G,C,P=L('cot_greedy.jsonl'),L('cot_s.jsonl'),L('pal_s.jsonl'); ids=sorted(G)
def vote(v):
    v=[p for p in v if p is not None]
    if not v: return None
    c=Counter(v); b=max(c.values()); return next(p for p in v if c[p]==b)
def sp(a,b):
    n=a+b; k=min(a,b); return 1.0 if n==0 else min(1,2*sum(comb(n,i) for i in range(k+1))/2**n)
gold={i:G[i]['gold'] for i in ids}
arm={'greedy':lambda i:G[i]['preds'][:1],'MIX10':lambda i:C[i]['preds'][:5]+P[i]['preds'][:5],
     'vote5C':lambda i:C[i]['preds'][:5],'vote5P':lambda i:P[i]['preds'][:5],
     'vote5C_longonly':lambda i:[p for p,s in zip(C[i]['preds'][:5],C[i]['samples'][:5]) if s['tokens']>=30]}
ok={a:{i:vote(f(i))==gold[i] for i in ids} for a,f in arm.items()}
def cmp(a,b,sub):
    fx=sum(ok[a][i] and not ok[b][i] for i in sub); br=sum(ok[b][i] and not ok[a][i] for i in sub)
    return f"{a}-{b}: n={len(sub)} {100*(fx-br)/len(sub):+.1f} ({fx}/{br}) p={sp(fx,br):.4f}"
long=[i for i in ids if G[i]['samples'][0]['tokens']>=30]; short=[i for i in ids if i not in long]
print(f"greedy no-reasoning (<30 tok): {len(short)}/200, acc {sum(G[i]['ok'] for i in short)}/{len(short)}; with reasoning acc {sum(G[i]['ok'] for i in long)}/{len(long)}")
cs=[s for i in ids for s in C[i]['samples']]
sh=[s for s in cs if s['tokens']<30]
print(f"CoT samples no-reasoning: {len(sh)}/{len(cs)}; acc short {sum(s['pred']==gold[i] for i in ids for s in C[i]['samples'] if s['tokens']<30)}/{len(sh)}, long {sum(s['pred']==gold[i] for i in ids for s in C[i]['samples'] if s['tokens']>=30)}/{len(cs)-len(sh)}")
for a,b in [('MIX10','vote5C'),('MIX10','vote5P'),('vote5C_longonly','greedy'),('MIX10','vote5C_longonly')]: print(cmp(a,b,ids))
for sub,nm in [(long,'greedy reasoned'),(short,'greedy skipped')]:
    print(nm, cmp('MIX10','greedy',sub))
for a in ok: print(a, f"acc={sum(ok[a].values())/200:.3f}")
