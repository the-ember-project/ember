# Exploratory (session #9): difficulty of the questions each arm's DPO pairs come from.
import gzip, json
from collections import Counter
def same(a, b): return a is not None and b is not None and abs(a - b) < 1e-6
def key(x): return round(x, 6)
rows = [json.loads(l) for l in gzip.open("gpu-output/gen_answers.jsonl.gz", "rt")]
def label(arm, r):
    ca, pa = r["cot"], r["pal"]
    if arm == "GOLD": return r["gold"]
    c = Counter(key(x) for x in (ca if arm == "COT" else ca + pa) if x is not None)
    if not c: return None
    lab, v = c.most_common(1)[0]
    if arm == "COT": return lab if v >= 5 else None
    return lab if v >= 7 and any(same(x, lab) for x in ca) and any(same(x, lab) for x in pa) else None
for arm in ["MIX", "COT", "GOLD"]:
    ks = []; wrong = 0
    for r in rows:
        lab = label(arm, r)
        if lab is None: continue
        nc = sum(same(x, lab) for x in r["cot"])
        if nc == 0 or nc == len(r["cot"]): continue
        ks.append(sum(same(x, r["gold"]) for x in r["cot"])); wrong += not same(lab, r["gold"])
    h = Counter(ks); n = len(ks)
    print(f"{arm}: pair_q={n} wrong_label={wrong} mean_cot_correct={sum(ks)/n:.2f}/8 "
          f"hard(<=4/8 correct)={sum(v for k,v in h.items() if k<=4)/n:.1%} easy(>=7/8)={sum(v for k,v in h.items() if k>=7)/n:.1%}")

# Relaxed label-free variants (what could a next round use?)
def relaxed(r, thr, need_both):
    ca, pa = r["cot"], r["pal"]
    c = Counter(key(x) for x in ca + pa if x is not None)
    if not c: return None
    lab, v = c.most_common(1)[0]
    if v < thr: return None
    if need_both and not (any(same(x, lab) for x in ca) and any(same(x, lab) for x in pa)): return None
    return lab
for thr in (5, 6, 7):
    ks = []; wrong = 0
    for r in rows:
        lab = relaxed(r, thr, True)
        if lab is None: continue
        nc = sum(same(x, lab) for x in r["cot"])
        if nc == 0 or nc == len(r["cot"]): continue
        ks.append(sum(same(x, r["gold"]) for x in r["cot"])); wrong += not same(lab, r["gold"])
    n = len(ks)
    print(f"MIX thr>={thr}/13: pair_q={n} precision={1-wrong/n:.1%} hard(<=4/8)={sum(k<=4 for k in ks)/n:.1%} hard_q={sum(k<=4 for k in ks)}")
