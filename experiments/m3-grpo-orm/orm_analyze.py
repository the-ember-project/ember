"""Verdicts for m3-grpo-orm (rules fixed in README before the run). Usage: python orm_analyze.py [outdir]"""
import gzip, json, math, os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "gpu-output")
REP = os.path.join(H, "..", "m3-grpo-rep", "gpu-output")

def sign_p(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
def opn(p): return gzip.open(p, "rt") if p.endswith(".gz") else open(p)
def find(d, name):
    for ext in (".gz", ""):
        p = os.path.join(d, name + ext)
        if os.path.exists(p): return p
def load_eval(d, arm):
    p = find(d, f"eval_{arm}.jsonl")
    if not p: return None, None
    m, b = {}, {}
    for l in opn(p):
        r = json.loads(l)
        if "idx" in r: m[r["idx"]] = r
        else: b[r["bbh_idx"]] = r
    return m, b
def cmp(x, y, items, lab):
    if not items: print(f"  {lab}: no items"); return 0.0, 1.0
    f = sum(x[i] and not y[i] for i in items); b = sum(y[i] and not x[i] for i in items)
    d = 100 * (sum(x[i] for i in items) - sum(y[i] for i in items)) / len(items)
    print(f"  {lab}: {d:+.1f} ({f}/{b}, p={sign_p(f, b):.3g}, n={len(items)})"); return d, sign_p(f, b)
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def wvote(ans, w):  # weighted plurality, ties to the earliest (math-verify rule)
    s, first = defaultdict(float), {}
    for j, (v, x) in enumerate(zip(ans, w)):
        if v is None: continue
        k = round(v, 4); s[k] += x; first.setdefault(k, j)
    return max(s, key=lambda k: (s[k], -first[k])) if s else None
def sig(x): return 1 / (1 + math.exp(-x))

E, B = {}, {}
for arm in ("BASE", "ORM1", "ORM2", "3B"):
    E[arm], B[arm] = load_eval(D, arm)
for arm in ("MAJ2", "MAJ3"):
    E[arm], B[arm] = load_eval(REP, arm)
if E["BASE"] is None: sys.exit(f"no eval_BASE in {D}")
items = sorted(E["BASE"])
acc = {a: {i: E[a][i]["correct"] for i in items} for a in E if E[a] is not None and set(items) <= set(E[a])}
print("n items", len(items), "arms", sorted(acc))
for a in sorted(acc):
    e = E[a]; print(f"{a}: acc {100*sum(acc[a].values())/len(items):.1f}  boxed {100*sum(e[i]['boxed'] for i in items)/len(items):.1f}"
                    f"  mean chars {sum(e[i]['len'] for i in items)/len(items):.0f}"
                    + (f"  BBH-date {100*sum(r['correct'] for r in B[a].values())/len(B[a]):.1f}" if B[a] else ""))
print("Validity: BASE 50-62, boxed>=88; 3B 60-75; >=90 steps")
for arm in ("ORM1", "ORM2"):
    p = os.path.join(D, f"train_{arm}.json")
    if os.path.exists(p):
        t = json.load(open(p)); st = t["stats"]
        def avg(k, sl): v = [s[k] for s in st[sl] if s.get(k) is not None]; return sum(v) / len(v) if v else float("nan")
        print(f"  {arm}: steps {t['steps']}  mem {t['gpu_mem_gb']} GB  reward batches {len(st)}")
        for k in ("gold_acc", "boxed", "rm_mean", "mean_chars", "rm_auroc"):
            print(f"    {k}: first10% {avg(k, slice(0, max(1, len(st)//10))):.3f}  last10% {avg(k, slice(-max(1, len(st)//10), None)):.3f}")
        print(f"    rm non-finite total {sum(s['rm_nonfinite'] for s in st)}")
print("O1 / O2 (greedy, 1000 items):")
g = []
for arm in ("ORM1", "ORM2"):
    if arm in acc:
        d, p = cmp(acc[arm], acc["BASE"], items, f"{arm} - BASE"); g.append((d, p))
        print(f"  O1 {arm}: {'PASS' if d >= 2.0 and p < 0.05 else 'FAIL'}")
        bx = [i for i in items if E["BASE"][i]["boxed"]]; cmp(acc[arm], acc["BASE"], bx, f"{arm} - BASE on BASE-boxed (expl)")
if len(g) == 2:
    m = (g[0][0] + g[1][0]) / 2
    print(f"  O2 mean {m:+.2f}: {'PASS' if m >= 3.0 and all(p < 0.05 for _, p in g) else 'FAIL'}")
if all(a in acc for a in ("ORM1", "ORM2", "MAJ2", "MAJ3")):
    print("O3 (ORM vs MAJ reward, cross-job, same items):")
    d1, _ = cmp(acc["ORM1"], acc["MAJ2"], items, "ORM1 - MAJ2"); d2, _ = cmp(acc["ORM2"], acc["MAJ3"], items, "ORM2 - MAJ3")
    print(f"  O3 mean {(d1 + d2) / 2:+.2f}: {'PASS' if (d1 + d2) / 2 >= 1.5 else 'FAIL'}")
else: print("O3: not testable (missing arms)")
if "3B" in acc:
    cmp(acc["3B"], acc["BASE"], items, "3B greedy - BASE greedy (expl)")
S = {}
for arm in ("ORM1", "BASE"):
    gp, rp = find(D, f"gen_{arm}.jsonl"), find(D, f"rm_{arm}_rm06.json")
    if gp and rp:
        G = {r["idx"]: r for r in (json.loads(l) for l in opn(gp))}; R = json.load(open(rp))
        print(f"{arm} RM non-finite fp16 {R['n_nonfinite_fp16']} final {R['n_nonfinite_final']}")
        S[arm] = {i: ok(wvote(G[i]["ans"], [sig(x) for x in R["scores"][str(i)]]), G[i]["gold"]) for i in G}
        S[arm + "_SC8"] = {i: ok(wvote(G[i]["ans"], [1] * len(G[i]["ans"])), G[i]["gold"]) for i in G}
        S[arm + "_greedy"] = {i: ok(G[i]["ans"][0], G[i]["gold"]) for i in G}
if "ORM1" in S:
    si = sorted(S["ORM1"])
    for k in sorted(S): print(f"  {k}: {100*sum(S[k][i] for i in si)/len(si):.1f}  (n={len(si)})")
    if "3B" in acc and set(si) <= set(acc["3B"]):
        d, p = cmp(S["ORM1"], acc["3B"], si, "O4 WSC8(ORM1) - 3B greedy")
        print(f"  O4 (M5): {'PASS' if d >= -2.0 else 'FAIL'}")
    if "BASE" in S:
        d, p = cmp(S["ORM1"], S["BASE"], si, "O5 WSC8(ORM1) - WSC8(BASE)")
        print(f"  O5: {'PASS' if d >= 1.5 else 'FAIL'}")
        cmp(S["BASE"], S["BASE_SC8"], si, "WSC8(BASE) - SC8(BASE) (expl)")
