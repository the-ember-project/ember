"""Verdicts for m3-grpo-rep from per-item files. Usage: python rep_analyze.py [dir] (default gpu-output/)."""
import os, sys
import analyze as A

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-output")
A.D = D
E = {a: A.load(a) for a in ("BASE", "MAJ2", "MAJ3")}
E = {a: v for a, v in E.items() if v[0] is not None}
B = E["BASE"][0]
for a, (m, bb) in E.items():
    assert [r["idx"] for r in m] == [r["idx"] for r in B], a
    print(f"{a:5s} acc={100*sum(r['correct'] for r in m)/len(m):.1f} boxed={100*sum(r['boxed'] for r in m)/len(m):.1f} n={len(m)} "
          f"bbh_date={100*sum(r['correct'] for r in bb)/max(1,len(bb)):.1f}")
bmask = [r["boxed"] for r in B]
R, Rb = {}, {}
for a in ("MAJ2", "MAJ3"):
    if a not in E: print(a, "missing"); continue
    R[a] = A.paired(E[a][0], B); Rb[a] = A.paired(E[a][0], B, bmask)
    ok = R[a]["diff"] >= 2.0 and R[a]["p"] < 0.05
    print(f"R1 {a}-BASE", R[a], "PASS" if ok else "FAIL")
    print(f"   boxed-only {a}-BASE", Rb[a])
    bbd = 100 * (sum(r["correct"] for r in E[a][1]) - sum(r["correct"] for r in E["BASE"][1])) / max(1, len(E["BASE"][1]))
    print(f"   drift BBH-date {a}-BASE {bbd:+.1f}", "ok" if bbd >= -4 else "DRIFT")
if len(R) == 2:
    mean = (R["MAJ2"]["diff"] + R["MAJ3"]["diff"]) / 2
    print(f"R2 mean {mean:+.2f}", "PASS" if mean >= 3.0 and all(r["p"] < 0.05 for r in R.values()) else "FAIL")
    mb = sum((Rb[a] or {"diff": 0})["diff"] for a in ("MAJ2", "MAJ3")) / 2
    print(f"R3 boxed-only mean {mb:+.2f}", "PASS" if mb >= 1.5 else "FAIL")
    print("expl MAJ2-MAJ3", A.paired(E["MAJ2"][0], E["MAJ3"][0]))
acc = 100 * sum(r["correct"] for r in B) / len(B); bx = 100 * sum(bmask) / len(B)
print(f"validity: BASE acc {acc:.1f} in 50-62: {50 <= acc <= 62}; boxed {bx:.1f} >= 88: {bx >= 88}; n={len(B)}")
for a in ("MAJ2", "MAJ3"):
    p = os.path.join(D, f"train_{a}.json")
    if os.path.exists(p):
        import json; t = json.load(open(p)); print(f"train {a}: steps={t['steps']} (>=90: {t['steps'] >= 90})")
