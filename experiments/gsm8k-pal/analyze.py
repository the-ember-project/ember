"""Paired PAL vs greedy CoT analysis on whatever PAL results exist (offset 0 and/or 100)."""
import json, math, os
B = "experiments/gsm8k-baseline/"; P = "experiments/gsm8k-pal/"
def load(p): return {r["idx"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
cot = {**load(B+"results_qwen1.5b_greedy_n100.jsonl"), **load(B+"results_qwen1.5b_greedy_off100_n200.jsonl")}
pal = {**load(P+"results_pal_n100.jsonl"), **load(P+"results_pal_off100_n200.jsonl")}
ids = [i for i in pal if i in cot]; n = len(ids)
def sign(b, c):
    k = min(b, c); return min(1.0, 2*sum(math.comb(b+c, i) for i in range(k+1))/2**(b+c)) if b+c else 1.0
C = sum(cot[i]["ok"] for i in ids); Pk = sum(pal[i]["ok"] for i in ids)
b = sum(1 for i in ids if pal[i]["ok"] and not cot[i]["ok"]); c = sum(1 for i in ids if cot[i]["ok"] and not pal[i]["ok"])
print(f"n={n} CoT={C/n:.3f} PAL={Pk/n:.3f} diff={100*(Pk-C)/n:+.1f}pts fixed={b} broken={c} sign-test p={sign(b,c):.3f}")
err = sum(1 for i in ids if not pal[i]["exec_ok"] or pal[i]["pred"] is None)
print(f"exec errors/no output: {err} ({100*err/n:.1f}%)  tokens/q PAL={sum(pal[i]['tokens'] for i in ids)/n:.0f} CoT={sum(cot[i]['tokens'] for i in ids)/n:.0f}")
hyb = sum((cot[i]["ok"] if (not pal[i]["exec_ok"] or pal[i]["pred"] is None) else pal[i]["ok"]) for i in ids)
print(f"[exploratory] hybrid (PAL, fallback CoT on error) = {hyb/n:.3f}")
agree = [i for i in ids if pal[i]["pred"] is not None and cot[i]["pred"] is not None and abs(pal[i]["pred"]-cot[i]["pred"]) < 1e-4]
print(f"[exploratory] agree on {len(agree)}/{n}; acc when agree={sum(cot[i]['ok'] for i in agree)/max(len(agree),1):.3f}; "
      f"either correct (oracle of 2)={sum(cot[i]['ok'] or pal[i]['ok'] for i in ids)/n:.3f}")
