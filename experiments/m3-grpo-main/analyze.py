"""Verdicts for m3-grpo-main from per-item files. Usage: python analyze.py [dir] (default gpu-output/)."""
import gzip, json, math, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-output")

def load(arm, tag=""):
    for p in (f"{D}/{tag}eval_{arm}.jsonl.gz", f"{D}/{tag}eval_{arm}.jsonl"):
        if os.path.exists(p):
            rows = [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
            return [r for r in rows if "idx" in r], [r for r in rows if "bbh_idx" in r]
    return None, None

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

def paired(x, y, mask=None):
    xs = [r["correct"] for r in x]; ys = [r["correct"] for r in y]
    if mask: xs = [v for v, m in zip(xs, mask) if m]; ys = [v for v, m in zip(ys, mask) if m]
    if not xs: return None
    f = sum(p and not q for p, q in zip(xs, ys)); b = sum(q and not p for p, q in zip(xs, ys))
    return dict(diff=round(100 * (sum(xs) - sum(ys)) / len(xs), 2), fixed=f, broken=b, p=sign_p(f, b), n=len(xs))

def main():
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    E = {a: load(a, tag) for a in ("BASE", "GOLD", "MAJ", "RANDOM")}
    E = {a: v for a, v in E.items() if v[0] is not None}
    B = E["BASE"][0]
    for a, (m, bb) in E.items():
        assert [r["idx"] for r in m] == [r["idx"] for r in B], a
        print(f"{a:7s} acc={100*sum(r['correct'] for r in m)/len(m):.1f} boxed={100*sum(r['boxed'] for r in m)/len(m):.1f} "
              f"n={len(m)} bbh_date={100*sum(r['correct'] for r in bb)/max(1,len(bb)):.1f} meanlen={sum(r['len'] for r in m)/len(m):.0f}")
    bmask = [r["boxed"] for r in B]
    def show(name, x, y, mask=None, thr=None):
        if x not in E or y not in E: print(name, "missing arm"); return
        r = paired(E[x][0], E[y][0], mask); print(name, r, "" if thr is None or r is None else thr(r))
    ok = lambda r: "PASS" if r["diff"] >= 2.0 and r["p"] < 0.05 else "FAIL"
    show("H1 GOLD-BASE", "GOLD", "BASE", thr=ok)
    show("H2 MAJ-BASE", "MAJ", "BASE", thr=ok)
    show("H3 GOLD-RANDOM", "GOLD", "RANDOM", thr=ok)
    show("H4 RANDOM-BASE", "RANDOM", "BASE", thr=lambda r: "PASS" if abs(r["diff"]) < 2.0 else "FAIL")
    show("H5 GOLD-BASE on BASE-boxed", "GOLD", "BASE", mask=bmask, thr=lambda r: "PASS" if r["diff"] >= 1.5 else "FAIL")
    show("expl MAJ-BASE on BASE-boxed", "MAJ", "BASE", mask=bmask)
    show("expl MAJ-RANDOM", "MAJ", "RANDOM")
    if "GOLD" in E:
        x = [r["correct"] for r in E["GOLD"][1]]; y = [r["correct"] for r in E["BASE"][1]]
        if x: print("drift BBH-date GOLD-BASE", round(100 * (sum(x) - sum(y)) / len(x), 2), "PASS" if (sum(x) - sum(y)) / len(x) >= -0.04 else "FAIL")
    acc = 100 * sum(r["correct"] for r in B) / len(B); bx = 100 * sum(bmask) / len(B)
    print(f"validity: BASE acc {acc:.1f} in 50-62: {50 <= acc <= 62}; BASE boxed {bx:.1f} >= 88: {bx >= 88}; n={len(B)}")
    for a in ("GOLD", "MAJ", "RANDOM"):
        p = f"{D}/{tag}train_{a}.json"
        if not os.path.exists(p): print("train", a, "missing"); continue
        t = json.load(open(p)); st = t["stats"]
        f = lambda k, s: round(sum(x[k] for x in s) / max(1, len(s)), 3)
        print(f"train {a}: steps={t['steps']} (>=90: {t['steps'] >= 90}) s/step={sorted(t['sec_per_step'])[len(t['sec_per_step'])//2]:.0f} "
              f"gold_acc first20={f('gold_acc', st[:20])} last20={f('gold_acc', st[-20:])} boxed last20={f('boxed', st[-20:])} "
              f"maj_prec={f('maj_precision', st)} maj_reward_acc={f('maj_reward_acc', st)}")

if __name__ == "__main__":
    main()
