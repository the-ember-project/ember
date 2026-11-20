"""Exploratory: training-free selectors over stored 1.5B MATH samples (math-mix7 rows + math-sc7 cs2), n=250."""
import json, math, os
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..")
B = {r["idx"]: r for r in (json.loads(l) for l in open(f"{R}/math-mix7/rows.jsonl") if l.strip())}
C = [json.loads(l) for l in open(f"{R}/math-sc7/cs2.jsonl") if l.strip()]
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def sign(f, b):
    n = f + b; k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
def wvote(items):  # items: (answer, weight, order); ties -> earliest
    s = defaultdict(float); first = {}
    for i, (a, w) in enumerate(items):
        k = key(a)
        if k is None: continue
        s[k] += w; first.setdefault(k, i)
    return max(s, key=lambda k: (s[k], -first[k])) if s else None
A = defaultdict(list)
for c in C:
    r = B[c["idx"]]; g = r["gold"]
    ans = [r["cot"]] + r["cs"] + c["cs2"]; tok = [r["cot_tok"]] + r["cs_tok"] + c["cs2_tok"]
    bx = [r["boxed"]] + r["cs_boxed"] + c["cs2_boxed"]
    A["SC7"].append(ok(wvote([(a, 1) for a in ans]), g))
    A["pass@7"].append(any(ok(a, g) for a in ans))
    A["SC7_boxedonly"].append(ok(wvote([(a, 1) for a, b in zip(ans, bx) if b]), g))
    A["W_1/tok"].append(ok(wvote([(a, 1 / t) for a, t in zip(ans, tok)]), g))
    A["W_1/sqrt"].append(ok(wvote([(a, 1 / math.sqrt(t)) for a, t in zip(ans, tok)]), g))
    A["SC7_tieshort"].append(ok(wvote([(a, 1 + 1e-6 / t) for a, t in zip(ans, tok)]), g))
    bt = [(t, a) for a, t, b in zip(ans, tok, bx) if b and key(a) is not None]
    A["shortest_boxed"].append(ok(min(bt)[1], g) if bt else False)
    A["MIX10"].append(ok(wvote([(a, 1) for a in ans + r["pal"]]), g))
    A["SC7+PALhalf"].append(ok(wvote([(a, 1) for a in ans] + [(p, .5) for p in r["pal"]]), g))
    A["SC7+PALtie"].append(ok(wvote([(a, 1) for a in ans] + [(p, .01) for p in r["pal"]]), g))
    # margin stats
n = len(C)
for k, v in A.items():
    f = sum(a and not b for a, b in zip(v, A["SC7"])); b = sum(b and not a for a, b in zip(v, A["SC7"]))
    print(f"{k:15s} {100*sum(v)/n:5.1f}  vs SC7 {100*(f-b)/n:+5.1f} ({f}/{b}) p={sign(f,b):.3g}")
# Where does SC7 miss pass@7? count of correct samples among 7 on those items
from collections import Counter
cc = Counter(); top = Counter()
for c in C:
    r = B[c["idx"]]; g = r["gold"]; ans = [r["cot"]] + r["cs"] + c["cs2"]
    if any(ok(a, g) for a in ans) and not ok(wvote([(a, 1) for a in ans]), g):
        k = sum(ok(a, g) for a in ans); cc[k] += 1
        m = Counter(key(a) for a in ans if key(a) is not None).most_common(1)[0][1]; top[(k, m)] += 1
print("SC7 misses with gold present: #correct-of-7 ->", dict(sorted(cc.items())))
print("(#correct, plurality size) ->", dict(sorted(top.items())))
