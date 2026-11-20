"""Exploratory (post-hoc): where does the self-judge fail? Run after analyze.py."""
import json, math, os
from collections import defaultdict
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")).read().split("A = defaultdict")[0])
rk, clus, conf = [], [], []
for s in S:
    r, c = B[s["idx"]], C[s["idx"]]; g = r["gold"]
    ans = [r["cot"]] + r["cs"] + c["cs2"]; L = [lo(p) for p in s["sv"]]
    cor = [ok(a, g) for a in ans]; sc = ok(wvote(ans, [1]*7), g)
    conf.append((sum(L)/7, sc))
    if any(cor) and not sc:
        order = sorted(range(7), key=lambda i: -L[i])
        rk.append(min(order.index(i) for i in range(7) if cor[i]) + 1)
        cm = [L[i] for i in range(7) if cor[i]]; wm = [L[i] for i in range(7) if not cor[i]]
        clus.append(sum(cm)/len(cm) > sum(wm)/len(wm))
print(f"SC7-miss-with-correct items: {len(rk)}; best-correct rank by self-score: "
      + ", ".join(f"r{k}:{rk.count(k)}" for k in range(1, 8)))
print(f"  mean score of correct samples > mean of wrong samples on {sum(clus)}/{len(clus)}")
pos = [x for x, y in conf if y]; neg = [x for x, y in conf if not y]
print(f"question-level AUROC (mean self-score predicts SC7 correct): {auroc(pos, neg):.3f}")
# vote agreement (top count / 7) as the baseline confidence signal
ag = []
for s in S:
    r, c = B[s["idx"]], C[s["idx"]]; ans = [r["cot"]] + r["cs"] + c["cs2"]
    cnt = defaultdict(int)
    for a_ in ans:
        if key(a_) is not None: cnt[key(a_)] += 1
    ag.append(max(cnt.values()) / 7 if cnt else 0)
pos = [a_ for a_, (_, y) in zip(ag, conf) if y]; neg = [a_ for a_, (_, y) in zip(ag, conf) if not y]
print(f"question-level AUROC of vote agreement: {auroc(pos, neg):.3f}")
comb = [a_ + 0.02 * x for a_, (x, _) in zip(ag, conf)]  # agreement, self-score as tie-break only
pos = [a_ for a_, (_, y) in zip(comb, conf) if y]; neg = [a_ for a_, (_, y) in zip(comb, conf) if not y]
print(f"question-level AUROC of agreement + self-score tie-break: {auroc(pos, neg):.3f}")
