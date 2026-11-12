# Exploratory (session #19): sequential PAL sampling, stop once the G3 vote cannot change. Usage: python adaptive.py rows.jsonl ...
import json,itertools,sys
from collections import Counter
def key(x): return x if (x is None or isinstance(x,str)) else round(float(x),4)
def vote(c,pals):
    v=Counter(key(x) for x in [c]+pals if x is not None)
    if not v: return key(c)
    top=max(v.values()); best=[a for a,n in v.items() if n==top]
    if key(c) in best: return key(c)
    if len(best)==1: return best[0]
    return next(key(x) for x in pals if key(x) in best)
for fn in sys.argv[1:]:
    R=[json.loads(l) for l in open(fn)]
    used=0;mism=0;ctok=0;ptok=0;atok=0
    for r in R:
        c=r['cot'];P=r['pal'];per=r['pal_tok']/len(P)
        full=vote(c,P)
        for k in range(len(P)+1):
            cur=P[:k];rem=len(P)-k
            opts=list({key(x) for x in [c]+cur if x is not None})+['N1','N2',None]
            outs={vote(c,cur+list(comb)) for comb in itertools.product(opts,repeat=rem)}
            if len(outs)==1: break
        used+=k; mism+= (vote(c,P[:k]+[None]*(3-k))!=full)
        ctok+=r['cot_tok'];ptok+=r['pal_tok'];atok+=k*per
    n=len(R)
    print(fn,'n',n,'mean PAL samples',round(used/n,2),'mismatch',mism,'cost full',round((ctok+ptok)/ctok,2),'x adaptive',round((ctok+atok)/ctok,2),'x')
