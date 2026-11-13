"""G3 vs SC4 at equal voter count on the 250 math-g3 questions."""
import json, math, sys
from collections import OrderedDict
def load(f): return {r["idx"]: r for r in (json.loads(l) for l in open(f) if l.strip())}
G, F, S = load("../math-g3/rows.jsonl"), load("../math-format3/rows.jsonl"), load(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")
def key(v): return None if v is None else round(v, 4)
def vote(c):  # plurality, ties to first (greedy) in insertion order
    s = OrderedDict()
    for v in c:
        k = key(v)
        if k is None: continue
        s[k] = s.get(k, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def sign(f, b):
    n = f + b; k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
ids = [i for i in G if i in F and i in S]; n = len(ids)
bx = sum(S[i]["cs2_boxed"] + S[i]["cs3_boxed"] for i in ids) / max(1, 2 * n)
print(f"n={n}  CS2/CS3 boxed: {100*bx:.1f}%")
A = {}
for i in ids:
    g, f, s = G[i], F[i], S[i]; gold = g["gold"]
    cs = [f["cs"], s["cs2"], s["cs3"]]
    for name, c in {"COT": [g["cot"]], "G3": [g["cot"]] + g["pal"][:3], "SC4": [g["cot"]] + cs,
                    "MIX7": [g["cot"]] + cs + g["pal"][:3]}.items():
        A.setdefault(name, []).append(ok(vote(c), gold))
    A.setdefault("G3stored", []).append(g["g3_ok"])
for k, v in A.items(): print(f"{k:9s} {100*sum(v)/n:.1f}%")
def cmp(x, y, lab):
    f = sum(a and not b for a, b in zip(A[x], A[y])); b = sum(b and not a for a, b in zip(A[x], A[y]))
    print(f"{lab}: {x}-{y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign(f,b):.4g}")
cmp("G3", "SC4", "H1/H2"); cmp("SC4", "COT", "expl"); cmp("G3", "COT", "expl"); cmp("MIX7", "G3", "expl")
print("G3 recomputed == stored:", sum(a == b for a, b in zip(A["G3"], A["G3stored"])), "/", n)
ct = sum(G[i]["cot_tok"] for i in ids) / n
pt = sum(G[i]["pal_tok"] for i in ids) / n
st = sum(F[i]["cs_tok"] + S[i]["cs2_tok"] + S[i]["cs3_tok"] for i in ids) / n
print(f"cost vs greedy tokens: G3 {(ct+pt)/ct:.2f}x  SC4 {(ct+st)/ct:.2f}x")
