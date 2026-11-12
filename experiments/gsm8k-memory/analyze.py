import json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")]
bank = [json.loads(l) for l in open(sys.argv[2] if len(sys.argv) > 2 else "bank.jsonl")]
def p_sign(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
n = len(rows); acc = {k: 100 * sum(r[k]["ok"] for r in rows) / n for k in "ZRX"}
mem = [b for b in bank if b["agree"]]
print(f"bank n={len(bank)} kept={len(mem)} precision={100*sum(b['gold_ok'] for b in mem)/max(len(mem),1):.1f}%"
      f" (all-bank CoT acc {100*sum(b['gold_ok'] for b in bank)/len(bank):.1f}%)")
print(f"n={n} " + " ".join(f"{k}={v:.1f}%" for k, v in acc.items()))
for k in "ZRX":
    print(f"  {k}: parsed {100*sum(r[k]['pred'] is not None for r in rows)/n:.1f}%  capped {100*sum(r[k].get('capped',False) for r in rows)/n:.1f}%  #### {100*sum(r[k]['hash'] for r in rows)/n:.1f}%  mean out tok {sum(r[k]['tok'] for r in rows)/n:.0f}"
          f"  mean prompt tok {sum(r[k]['ptok'] for r in rows)/n:.0f}")
for a, b in (("R", "Z"), ("R", "X"), ("X", "Z")):
    fx = sum(r[a]["ok"] and not r[b]["ok"] for r in rows); br = sum(r[b]["ok"] and not r[a]["ok"] for r in rows)
    print(f"{a}-{b}: {acc[a]-acc[b]:+.1f} (fixed {fx} / broken {br}) p={p_sign(fx, br):.4f}")
if n < 3: sys.exit()
s = sorted(rows, key=lambda r: r["near_sim"][-1]); t = n // 3
for name, part in (("low-sim", s[:t]), ("mid-sim", s[t:2*t]), ("high-sim", s[2*t:])):
    m = len(part)
    print(f"  {name} (top sim {part[0]['near_sim'][-1]:.2f}-{part[-1]['near_sim'][-1]:.2f}, n={m}): "
          + " ".join(f"{k}={100*sum(r[k]['ok'] for r in part)/m:.1f}" for k in "ZRX"))
