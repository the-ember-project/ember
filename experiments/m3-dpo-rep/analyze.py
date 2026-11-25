"""m3-dpo-rep verdicts (D1-D3 + validity) from per-question files. R1 = m3-stack run 3; R2/R3 = this job.
Usage: python analyze.py [gpu_output_dir]"""
import json, math, os, sys
H = os.path.dirname(os.path.abspath(__file__))
G = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "gpu-output")
DIRS = {"R1": os.path.join(H, "..", "m3-stack", "gpu-output"), "R2": os.path.join(G, "R2"), "R3": os.path.join(G, "R3")}

def sign_p(f, b):
    n = f + b; k = min(f, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def ld(p):
    return {d["idx"]: d for d in map(json.loads, open(p)) if "idx" in d}

res = {}
for r, d in DIRS.items():
    try:
        B, M = ld(os.path.join(d, "test_eval_BASE.jsonl")), ld(os.path.join(d, "test_eval_MIXR.jsonl"))
    except Exception as e:
        print(r, "missing:", e); continue
    ks = sorted(set(B) & set(M)); n = len(ks)
    f = sum(M[i]["correct"] and not B[i]["correct"] for i in ks); b = sum(B[i]["correct"] and not M[i]["correct"] for i in ks)
    accB = 100 * sum(B[i]["correct"] for i in ks) / n; accM = 100 * sum(M[i]["correct"] for i in ks) / n
    parsed = 100 * sum(M[i]["pred"] is not None for i in ks) / n
    try: steps = json.load(open(os.path.join(d, "train_log.json")))["MIXR"]["steps"]
    except Exception: steps = None
    res[r] = dict(n=n, BASE=round(accB, 2), MIXR=round(accM, 2), diff=round(accM - accB, 2), fixed=f, broken=b,
                  p=sign_p(f, b), parsed=round(parsed, 1), steps=steps, B=B)
    x = res[r]
    print(f"{r}: n={n} BASE {x['BASE']} MIXR {x['MIXR']} diff {x['diff']:+.2f} ({f}/{b}) p={x['p']:.4g} "
          f"parsed {parsed:.1f}% steps {steps}")
    if r != "R1":
        print(f"  D1 {r}: {'PASS' if x['diff'] >= 2.0 and x['p'] < 0.05 else 'FAIL'}; "
              f"validity BASE 67-72: {67 <= accB <= 72}, steps>=100: {steps is not None and steps >= 100}, parsed>=99: {parsed >= 99}")
if "R1" in res and "R2" in res:
    ag = sum(res["R1"]["B"][i]["correct"] == res["R2"]["B"][i]["correct"] for i in res["R2"]["B"] if i in res["R1"]["B"])
    print(f"BASE agreement R1 vs R2 per item: {ag}/{len(res['R2']['B'])}")
reps = [r for r in ("R1", "R2", "R3") if r in res]
if len(reps) == 3:
    mean = sum(res[r]["diff"] for r in reps) / 3; nsig = sum(res[r]["p"] < 0.05 for r in reps)
    print(f"D2 (M3 replicated): mean {mean:+.2f}, {nsig}/3 p<.05 -> {'PASS' if mean >= 3.0 and nsig >= 2 else 'FAIL'}")
    print(f"D3 (direction): R2 {res['R2']['diff']:+.2f}, R3 {res['R3']['diff']:+.2f} -> "
          f"{'PASS' if res['R2']['diff'] > 0 and res['R3']['diff'] > 0 else 'FAIL'}")
