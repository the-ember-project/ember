"""Verdicts for m3-math-stack from per-question files. Usage: python compare.py [dir] (default gpu-output/)."""
import gzip, json, math, os, sys
from collections import defaultdict
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-output")

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

def ev(name):
    m, bb = {}, {}
    for l in open(os.path.join(D, name)):
        r = json.loads(l)
        (m if "idx" in r else bb)[r.get("idx", r.get("bbh_idx"))] = r
    return m, bb

def g3(name):
    return {r["idx"]: r for r in map(json.loads, gzip.open(os.path.join(D, name), "rt"))}

def cmp(lab, x, y, thr=None):
    ks = sorted(set(x) & set(y)); f = sum(x[k] and not y[k] for k in ks); b = sum(y[k] and not x[k] for k in ks)
    d = 100 * (f - b) / max(1, len(ks)); p = sign_p(f, b)
    v = "" if thr is None else ("  PASS" if d >= thr[0] and (thr[1] is None or p < thr[1]) else "  FAIL")
    print(f"{lab}: {d:+.2f} (fixed {f} / broken {b}, n={len(ks)}) p={p:.4g}{v}")
    return d

B, Bb = ev("eval_BASE.jsonl"); C, Cb = ev("eval_COT.jsonl"); T, _ = ev("3b_eval_BASE.jsonl")
GB, GC = g3("g3_BASE.jsonl.gz"), g3("g3_COT.jsonl.gz")
acc = lambda m: 100 * sum(r["correct"] for r in m.values()) / max(1, len(m))
unb = lambda m: 100 * sum(not r["boxed"] for r in m.values()) / max(1, len(m))
pal = [x for g in (GB, GC) for r in g.values() for x in r["pal"]]
print(f"n={len(B)}  BASE {acc(B):.2f} (unboxed {unb(B):.1f}%)  COT {acc(C):.2f} (unboxed {unb(C):.1f}%)  "
      f"3B {acc(T):.2f} (unboxed {unb(T):.1f}%)  G3(BASE) {acc(GB):.2f}  G3(COT) {acc(GC):.2f}")
pe = 100 * sum(x is not None for x in pal) / max(1, len(pal))
print(f"VALIDITY: BASE unboxed<=5: {unb(B) <= 5}; BASE acc 45-68: {45 <= acc(B) <= 68}; "
      f"3B >= BASE+3: {acc(T) >= acc(B) + 3}; PAL exec {pe:.1f}% >=55: {pe >= 55}")
c = lambda m: {k: r["correct"] for k, r in m.items()}
cmp("H1 COT-BASE greedy (>=+2, p<.05)", c(C), c(B), (2, 0.05))
cmp("H2 G3(COT) - 3B greedy (>= -2)", c(GC), c(T), (-2, None))
cmp("H3 G3(COT) - COT greedy (>=+2, p<.05)", c(GC), c(C), (2, 0.05))
if Bb: cmp("H4 BBH-date COT-BASE (>= -3)", c(Cb), c(Bb), (-3, None))
print("--- exploratory")
cmp("G3(COT) - G3(BASE)", c(GC), c(GB)); cmp("G3(BASE) - BASE", c(GB), c(B)); cmp("3B - BASE", c(T), c(B))
ub = {k for k, r in B.items() if not r["boxed"]}
cmp("COT-BASE on BASE-unboxed items", {k: C[k]["correct"] for k in ub}, {k: B[k]["correct"] for k in ub})
cmp("COT-BASE on BASE-boxed items", {k: C[k]["correct"] for k in B if k not in ub}, {k: B[k]["correct"] for k in B if k not in ub})
lv = defaultdict(lambda: [0, 0, 0, 0])
for k, r in B.items():
    L = lv[r.get("level")]; L[0] += 1; L[1] += r["correct"]; L[2] += C[k]["correct"]; L[3] += T[k]["correct"]
for L, (n, b, co, t) in sorted(lv.items(), key=lambda z: str(z[0])):
    print(f"  {L}: n={n} BASE {100*b/n:.1f} COT {100*co/n:.1f} 3B {100*t/n:.1f}")
for f in ("summary.json", "3b_summary.json", "g3_summary.json", "train_stats.json"):
    p = os.path.join(D, f)
    if os.path.exists(p): print(f, json.dumps(json.load(open(p)).get("greedy_tok_per_q", json.load(open(p))))[:400])
