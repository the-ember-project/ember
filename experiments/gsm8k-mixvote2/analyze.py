"""Analysis for gsm8k-mixvote2: python experiments/gsm8k-mixvote2/analyze.py [dir suffix]"""
import json, math, os, sys
from collections import OrderedDict
D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
SUF = sys.argv[2] if len(sys.argv) > 2 else "off600_n300"
def load(f):
    p = os.path.join(D, f)
    return {r["idx"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
g, c2, pk = (load(f"{a}_{SUF}.jsonl") for a in ("cot_greedy", "cot_k2", "pal_k5"))
def eq(a, b): return a is not None and b is not None and abs(a - b) < 1e-4
def vote(c):
    s = OrderedDict()
    for v in c:
        if v is None: continue
        k = round(v, 4); s[k] = s.get(k, 0) + 1
    return max(s, key=lambda k: s[k]) if s else None
def sign(b, c):
    k = min(b, c); return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)) if b + c else 1.0
ids = sorted(i for i in g if i in c2 and i in pk); n = len(ids)
print(f"paired n={n} (greedy {len(g)}, CoT k2 {len(c2)}, PAL k5 {len(pk)})")
if not n: raise SystemExit
gold = {i: g[i]["gold"] for i in ids}
gt = sum(g[i]["tokens"] for i in ids) / n
ct = sum(c2[i]["tokens"] for i in ids) / n / len(c2[ids[0]]["preds"])
pt = sum(pk[i]["tokens"] for i in ids) / n / len(pk[ids[0]]["preds"])
print(f"tokens per sample: greedy CoT {gt:.0f}, CoT sample {ct:.0f}, PAL sample {pt:.0f}")
def arm(a, b): return lambda i: vote(c2[i]["preds"][:a] + pk[i]["preds"][:b])
S = {"greedy": (lambda i: g[i]["pred"], 1.0),
     "P1 vote(2CoT+5PAL)": (arm(2, 5), (2*ct+5*pt)/gt),
     "P2 vote(1CoT+4PAL)": (arm(1, 4), (ct+4*pt)/gt),
     "P3 vote(5PAL)": (arm(0, 5), 5*pt/gt),
     "[ref] vote(2CoT)": (arm(2, 0), 2*ct/gt)}
for a in range(3):
    for b in range(6):
        if a + b and f"{a}{b}" not in ("25", "14", "05", "20"): S[f"[expl] vote({a}CoT+{b}PAL)"] = (arm(a, b), (a*ct+b*pt)/gt)
ok = {k: {i: eq(f(i), gold[i]) for i in ids} for k, (f, _) in S.items()}
G = sum(ok["greedy"].values())
def cmp(x, y): b = sum(ok[x][i] and not ok[y][i] for i in ids); c = sum(ok[y][i] and not ok[x][i] for i in ids); return b, c
for k, (_, cost) in S.items():
    b, c = cmp(k, "greedy"); A = sum(ok[k].values())
    print(f"{k:28s} acc={A/n:.3f} diff={100*(A-G)/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f} cost={cost:.2f}x")
b, c = cmp("P1 vote(2CoT+5PAL)", "[ref] vote(2CoT)")
print(f"P4 P1 vs vote(2CoT): diff={100*(b-c)/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f}")
