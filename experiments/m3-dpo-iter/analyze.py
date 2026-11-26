"""m3-dpo-iter analysis: per chain, paired sign tests on the full GSM8K test (from per-question files).
Usage: python analyze.py [root]   (root defaults to gpu-output/)"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gpu-output")

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

def load(fn):
    d = {}
    for l in open(fn):
        r = json.loads(l)
        if "idx" in r: d[r["idx"]] = r
    return d

def cmp(x, y, nx, ny):
    ks = sorted(set(x) & set(y)); f = sum(x[k]["correct"] and not y[k]["correct"] for k in ks)
    b = sum(y[k]["correct"] and not x[k]["correct"] for k in ks)
    d = 100 * (f - b) / len(ks)
    print(f"  {nx} - {ny}: {d:+.2f} ({f}/{b}) p={sign_p(f, b):.4g} n={len(ks)}")
    return d, sign_p(f, b)

res = {}
for tag in ("A", "B"):
    d = os.path.join(root, tag)
    try:
        B = load(os.path.join(d, "test_r1_eval_BASE.jsonl")); R1 = load(os.path.join(d, "test_r1_eval_MIXR.jsonl"))
    except FileNotFoundError:
        print(tag, "missing round-1 eval"); continue
    acc = lambda m: 100 * sum(r["correct"] for r in m.values()) / len(m)
    parsed = lambda m: 100 * sum(r["pred"] is not None for r in m.values()) / len(m)
    print(f"chain {tag}: BASE {acc(B):.2f} R1 {acc(R1):.2f} (parsed {parsed(B):.1f}/{parsed(R1):.1f})")
    r = dict(r1=cmp(R1, B, "R1", "BASE"))
    for st in ("1", "2"):
        try: print(f"  train_stats round {st}:", open(os.path.join(d, f"train_stats_r{st}.json")).read().replace("\n", "")[:400])
        except FileNotFoundError: pass
    try:
        M = load(os.path.join(d, "test_r2_eval_BASE.jsonl")); R2 = load(os.path.join(d, "test_r2_eval_MIXR.jsonl"))
        print(f"  R1m {acc(M):.2f} R2 {acc(R2):.2f} (parsed {parsed(M):.1f}/{parsed(R2):.1f}); merge check |R1m-R1| = {abs(acc(M)-acc(R1)):.2f}")
        r["r2"] = cmp(R2, M, "R2", "R1m"); r["tot"] = cmp(R2, B, "R2", "BASE")
        L = lambda m: sum(x["len"] for x in m.values()) / len(m)
        print(f"  mean response chars: BASE {L(B):.0f} R1 {L(R1):.0f} R1m {L(M):.0f} R2 {L(R2):.0f}")
    except FileNotFoundError:
        print("  round-2 eval missing")
    res[tag] = r

def verdict(name, ok): print(f"{name}: {'PASS' if ok else 'FAIL'}")
if all("r2" in res.get(t, {}) for t in ("A", "B")):
    g2 = [res[t]["r2"][0] for t in ("A", "B")]; gt = [res[t]["tot"][0] for t in ("A", "B")]
    verdict(f"I1 mean R2-R1m {sum(g2)/2:+.2f} (>= +1.0, both > 0)", sum(g2) / 2 >= 1.0 and min(g2) > 0)
    verdict(f"I2 both R2-R1m p<.05 ({res['A']['r2'][1]:.3g}, {res['B']['r2'][1]:.3g})", max(res[t]["r2"][1] for t in ("A", "B")) < 0.05)
    verdict(f"I3 mean R2-BASE {sum(gt)/2:+.2f} (>= +5.0)", sum(gt) / 2 >= 5.0)
if all("r1" in res.get(t, {}) for t in ("A", "B")):
    g1 = [res[t]["r1"][0] for t in ("A", "B")]
    verdict(f"I4 mean R1-BASE {sum(g1)/2:+.2f} (>= +2.0, both > 0)", sum(g1) / 2 >= 2.0 and min(g1) > 0)
