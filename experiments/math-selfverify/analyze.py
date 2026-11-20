"""Verdicts for math-selfverify (rules fixed in README before the run)."""
import json, math, os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..")
B = {r["idx"]: r for r in (json.loads(l) for l in open(f"{R}/math-mix7/rows.jsonl") if l.strip())}
C = {r["idx"]: r for r in (json.loads(l) for l in open(f"{R}/math-sc7/cs2.jsonl") if l.strip())}
S = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "sv_tails.jsonl")) if l.strip()]
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def sign(f, b):
    n = f + b; k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
def lo(p): p = min(max(p, 1e-9), 1 - 1e-9); return math.log(p / (1 - p))
def wvote(ans, w, tb=None):  # weighted plurality; ties -> higher tb (if given), else earliest
    s = defaultdict(float); first = {}; best = defaultdict(lambda: -1e18)
    for i, (a, x) in enumerate(zip(ans, w)):
        k = key(a)
        if k is None: continue
        s[k] += x; first.setdefault(k, i)
        if tb is not None: best[k] = max(best[k], tb[i])
    if not s: return None
    return max(s, key=lambda k: (round(s[k], 9), best[k] if tb is not None else 0, -first[k]))
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
A = defaultdict(list); pos, neg, wq = [], [], []
for s in S:
    r, c = B[s["idx"]], C[s["idx"]]; g = r["gold"]
    ans = [r["cot"]] + r["cs"] + c["cs2"]; L = [lo(p) for p in s["sv"]]
    cor = [ok(a, g) for a in ans]
    for x, y in zip(L, cor): (pos if y else neg).append(x)
    qp = [x for x, y in zip(L, cor) if y]; qn = [x for x, y in zip(L, cor) if not y]
    if qp and qn: wq.append(auroc(qp, qn))
    med = sorted(L)[3]
    A["SC7"].append(ok(wvote(ans, [1] * 7), g))
    A["SVW"].append(ok(wvote(ans, [2 / (1 + math.exp(-(x - med))) for x in L]), g))
    A["SVtie"].append(ok(wvote(ans, [1] * 7, tb=L), g))
    v = [(x, -i, a) for i, (x, a) in enumerate(zip(L, ans)) if key(a) is not None]
    A["BoN"].append(ok(max(v)[2], g) if v else False)
    A["pass@7"].append(any(cor))
n = len(S)
print(f"n={n}  mean P(Yes) {sum(sum(s['sv']) for s in S)/(7*n):.3f}  min Yes+No mass {min(min(s['mass']) for s in S):.3f}")
print(f"H1 pooled AUROC (logit sv, sample correct) = {auroc(pos, neg):.3f}  (pos {len(pos)}, neg {len(neg)}); "
      f"within-question mean AUROC = {sum(wq)/max(1,len(wq)):.3f} over {len(wq)} mixed questions")
for k, v in A.items():
    f = sum(a and not b for a, b in zip(v, A["SC7"])); b = sum(b and not a for a, b in zip(v, A["SC7"]))
    print(f"{k:6s} {100*sum(v)/n:5.1f}  vs SC7 {100*(f-b)/n:+5.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
