"""m3-dpo-label analysis from per-question files. Usage: python analyze.py [root] (default gpu-output/)."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gpu-output")

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

def load(fn):
    d = {}
    if not os.path.exists(fn): return None
    for l in open(fn):
        r = json.loads(l)
        if "idx" in r: d[r["idx"]] = r
    return d

M = {"BASE": load(os.path.join(root, "G0", "test_eval_BASE.jsonl")),
     "MIXR": load(os.path.join(root, "G0", "test_eval_MIXR.jsonl")),
     "T3Da": load(os.path.join(root, "G0", "test_eval_T3D.jsonl")),
     "GOLD": load(os.path.join(root, "G1", "test_eval_GOLD.jsonl")),
     "T3Db": load(os.path.join(root, "G1", "test_eval_T3D.jsonl")),
     "3B": load(os.path.join(HERE, "..", "m3-stack", "gpu-output", "test3b_eval_BASE.jsonl"))}
M = {k: v for k, v in M.items() if v}
acc = lambda m: 100 * sum(r["correct"] for r in m.values()) / len(m)
for k, m in M.items():
    print(f"{k:5s} acc {acc(m):.2f} parsed {100*sum(r['pred'] is not None for r in m.values())/len(m):.1f}% "
          f"chars {sum(r['len'] for r in m.values())/len(m):.0f} n={len(m)}")
for d in ("G0", "G1"):
    for fn in ("train_stats_A.json", "train_stats_B.json", "train_log_A.json", "train_log_B.json"):
        p = os.path.join(root, d, fn)
        if os.path.exists(p): print(d, fn, open(p).read().replace("\n", "").replace(" ", "")[:500])

def cmp(x, y):
    if x not in M or y not in M: return None
    ks = sorted(set(M[x]) & set(M[y]))
    f = sum(M[x][k]["correct"] and not M[y][k]["correct"] for k in ks)
    b = sum(M[y][k]["correct"] and not M[x][k]["correct"] for k in ks)
    d = 100 * (f - b) / len(ks); p = sign_p(f, b)
    print(f"  {x} - {y}: {d:+.2f} ({f}/{b}) p={p:.4g}")
    return d, p

print("paired comparisons:")
R = {(x, y): cmp(x, y) for x, y in [("MIXR", "BASE"), ("GOLD", "BASE"), ("T3Da", "BASE"), ("T3Db", "BASE"),
                                     ("GOLD", "MIXR"), ("T3Da", "MIXR"), ("T3Db", "MIXR"), ("T3Da", "T3Db"),
                                     ("T3Da", "3B"), ("T3Db", "3B"), ("GOLD", "3B")]}
V = lambda name, ok: print(f"{name}: {'PASS' if ok else 'FAIL'}")
if R[("GOLD", "MIXR")]: d, p = R[("GOLD", "MIXR")]; V(f"L1 GOLD-MIXR {d:+.2f} p={p:.3g} (>= +2, p<.05)", d >= 2 and p < .05)
if R[("T3Da", "MIXR")] and R[("T3Db", "MIXR")]:
    g = [R[("T3Da", "MIXR")][0], R[("T3Db", "MIXR")][0]]
    V(f"L2 mean T3D-MIXR {sum(g)/2:+.2f} (>= +2, both > 0)", sum(g) / 2 >= 2 and min(g) > 0)
t = [R[k] for k in (("T3Da", "BASE"), ("T3Db", "BASE")) if R[k]]
if t: V(f"L3 T3D-BASE {[round(x[0], 2) for x in t]} (each >= +3, p<.05; {len(t)} seeds)", len(t) == 2 and all(x[0] >= 3 and x[1] < .05 for x in t))
if "3B" in M and any(k in M for k in ("T3Da", "T3Db")):
    best = max(acc(M[k]) for k in ("T3Da", "T3Db") if k in M)
    V(f"L4 best T3D {best:.2f} vs 3B greedy {acc(M['3B']):.2f} - 2", best >= acc(M["3B"]) - 2)
if "BASE" in M: print(f"validity: BASE in 67-72: {67 <= acc(M['BASE']) <= 72}")
