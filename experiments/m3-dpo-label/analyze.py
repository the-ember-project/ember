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

# ---- Stage 4 (session #48): mix7 vote per arm, V1-V3 ----
import gzip
def loadz(fn):
    if not os.path.exists(fn): return None
    return {r["idx"]: r for r in (json.loads(l) for l in gzip.open(fn, "rt"))}
X = {"BASE": loadz(os.path.join(root, "G0", "test_mix_BASE.jsonl.gz")), "MIXR": loadz(os.path.join(root, "G0", "test_mix_MIXR.jsonl.gz")),
     "T3Da": loadz(os.path.join(root, "G0", "test_mix_T3D.jsonl.gz")), "GOLD": loadz(os.path.join(root, "G1", "test_mix_GOLD.jsonl.gz")),
     "T3Db": loadz(os.path.join(root, "G1", "test_mix_T3D.jsonl.gz"))}
X = {k: v for k, v in X.items() if v}
TOK = {}
for d, arms in (("G0", {"BASE": "BASE", "MIXR": "MIXR", "T3D": "T3Da"}), ("G1", {"GOLD": "GOLD", "T3D": "T3Db"})):
    p = os.path.join(root, d, "test_mix_summary.json")
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            if k in arms: TOK[arms[k]] = v
CAP = 5 * 275.4
if X:
    print("mix7 (2 CoT + 5 PAL, T=0.7):")
    for k, m in X.items():
        pe = sum(x is not None for r in m.values() for x in r["pal"]) / (5 * len(m))
        print(f"  {k:5s} mix7 {acc(m):.2f} n={len(m)} pal_exec {100*pe:.1f}% tok/q {TOK.get(k, {}).get('mix7_tok_per_q')} "
              f"g3(expl) {TOK.get(k, {}).get('g3_acc')}")
    def cmpx(A, x, B, y):
        ks = sorted(set(A) & set(B))
        f = sum(A[k]["correct"] and not B[k]["correct"] for k in ks); b = sum(B[k]["correct"] and not A[k]["correct"] for k in ks)
        d = 100 * (f - b) / len(ks); p = sign_p(f, b); print(f"  {x} - {y}: {d:+.2f} ({f}/{b}) p={p:.4g}"); return d, p
    t = [k for k in ("T3Da", "T3Db") if k in X]
    if len(t) == 2:
        ok = all(acc(X[k]) >= 83.2 and (TOK.get(k, {}).get("mix7_tok_per_q") or 1e9) <= CAP for k in t)
        V(f"V1 T3D mix7 {[round(acc(X[k]), 2) for k in t]} (both >= 83.2, tok <= {CAP:.0f})", ok)
    if "MIXR" in X and "BASE" in X:
        d, p = cmpx(X["MIXR"], "mix7 MIXR", X["BASE"], "mix7 BASE"); V("V2 (>= +2, p<.05)", d >= 2 and p < .05)
    if "BASE" in X and "BASE" in M:
        d, p = cmpx(X["BASE"], "mix7 BASE", M["BASE"], "greedy BASE"); V("V3 (>= +5, p<.05)", d >= 5 and p < .05)
    for k in X:
        if k != "BASE" and k in M: cmpx(X[k], f"mix7 {k}", M[k], f"greedy {k}")
