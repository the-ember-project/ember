"""Analysis for gsm8k-mixvote. Run from anywhere: python experiments/gsm8k-mixvote/analyze.py"""
import json, math, os
from collections import OrderedDict
D = os.path.dirname(os.path.abspath(__file__))
def load(f):
    p = os.path.join(D, f)
    return {r["idx"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
g, ck, pg, pk = (load(f"{a}_off300_n300.jsonl") for a in ("cot_greedy", "cot_k5", "pal_greedy", "pal_k5"))
def eq(a, b): return a is not None and b is not None and abs(a - b) < 1e-4
def vote(c):  # c: list of (pred, weight); ties -> earliest candidate
    s = OrderedDict()
    for v, w in c:
        if v is None: continue
        k = round(v, 4); s[k] = s.get(k, 0) + w
    return max(s, key=lambda k: s[k]) if s else None   # max returns first maximal key = earliest
def sign(b, c):
    k = min(b, c); return min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) / 2 ** (b + c)) if b + c else 1.0
ids = sorted(i for i in g if i in ck and i in pg and i in pk); n = len(ids)
print(f"paired questions with all 4 arms: n={n} (greedy CoT {len(g)}, CoT k5 {len(ck)}, PAL greedy {len(pg)}, PAL k5 {len(pk)})")
if not n: raise SystemExit
gold = {i: g[i]["gold"] for i in ids}
S = {
    "greedy": lambda i: g[i]["pred"],
    "S1 maj5CoT+PAL(w1.5)": lambda i: vote([(v, 1) for v in ck[i]["preds"]] + [(pg[i]["pred"], 1.5)]),
    "S2 maj(5CoT+5PAL)": lambda i: vote([(v, 1) for v in ck[i]["preds"]] + [(v, 1) for v in pk[i]["preds"]]),
    "S3 maj5CoT": lambda i: vote([(v, 1) for v in ck[i]["preds"]]),
    "[expl] PAL greedy": lambda i: pg[i]["pred"],
    "[expl] maj5PAL": lambda i: vote([(v, 1) for v in pk[i]["preds"]]),
    "[expl] maj5CoT+PAL(w1)": lambda i: vote([(v, 1) for v in ck[i]["preds"]] + [(pg[i]["pred"], 1)]),
    "[expl] maj4CoT+PAL(w1.5) (~4.5x)": lambda i: vote([(v, 1) for v in ck[i]["preds"][:4]] + [(pg[i]["pred"], 1.5)]),
    "[expl] greedyCoT+PALgreedy+maj5CoT": lambda i: vote([(g[i]["pred"], 1), (pg[i]["pred"], 1)] + [(v, 1) for v in ck[i]["preds"]]),
}
ok = {k: {i: eq(f(i), gold[i]) for i in ids} for k, f in S.items()}
tok = {"greedy": sum(g[i]["tokens"] for i in ids) / n}
tok["S1 maj5CoT+PAL(w1.5)"] = sum(ck[i]["tokens"] + pg[i]["tokens"] for i in ids) / n
tok["S2 maj(5CoT+5PAL)"] = sum(ck[i]["tokens"] + pk[i]["tokens"] for i in ids) / n
tok["S3 maj5CoT"] = sum(ck[i]["tokens"] for i in ids) / n
G = sum(ok["greedy"].values())
for k in S:
    A = sum(ok[k].values()); b = sum(ok[k][i] and not ok["greedy"][i] for i in ids); c = sum(ok["greedy"][i] and not ok[k][i] for i in ids)
    t = f" tokens/q={tok[k]:.0f} ({tok[k]/tok['greedy']:.1f}x)" if k in tok else ""
    print(f"{k:36s} acc={A/n:.3f} diff={100*(A-G)/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f}{t}")
b = sum(ok["S1 maj5CoT+PAL(w1.5)"][i] and not ok["S3 maj5CoT"][i] for i in ids)
c = sum(ok["S3 maj5CoT"][i] and not ok["S1 maj5CoT+PAL(w1.5)"][i] for i in ids)
print(f"S1 vs S3: diff={100*(b-c)/n:+.1f} fixed={b} broken={c} p={sign(b,c):.3f}")
pass5 = sum(any(eq(v, gold[i]) for v in ck[i]["preds"]) for i in ids) / n
print(f"[expl] pass@5 CoT={pass5:.3f}; pass@(5CoT+5PAL)={sum(any(eq(v, gold[i]) for v in ck[i]['preds'] + pk[i]['preds']) for i in ids)/n:.3f}")
