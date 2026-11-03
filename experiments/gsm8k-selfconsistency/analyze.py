"""Paired analysis: greedy vs self-consistency (maj@1/3/5), two scorers applied identically to both arms."""
import json, re, math, sys
from collections import Counter
D = "experiments/"
g = {r["idx"]: r for r in map(json.loads, open(D+"gsm8k-baseline/results_qwen1.5b_greedy_n100.jsonl"))}
s = {r["idx"]: r for r in map(json.loads, open(D+"gsm8k-selfconsistency/results_sc_k5_t0.7_n100.jsonl"))}
assert g.keys() == s.keys()
NUM = r"-?[\d,]*\.?\d+"
def raw(t):  # original eval.py scorer
    m = re.findall(r"####\s*\$?(" + NUM + ")", t) or re.findall(NUM, t)
    try: return float(m[-1].replace(",", "")) if m else None
    except ValueError: return None
def strict(t):  # v2: '####' > \boxed{} > 'answer is' > last number (v2); strips $ and commas
    for pat in [r"####\s*\$?\s*(" + NUM + ")", r"\\boxed\{\s*\\?\$?\s*(" + NUM + ")",
                r"answer is[^\d\-]{0,20}(" + NUM + ")"]:
        m = re.findall(pat, t, re.I)
        if m: break
    else: m = re.findall(NUM, t)
    try: return float(m[-1].replace(",", "")) if m else None
    except ValueError: return None
def vote(ps):
    c = Counter(p for p in ps if p is not None); return c.most_common(1)[0][0] if c else None
def eq(p, gold): return p is not None and abs(p - gold) < 1e-6
def mcnemar(b, c):  # exact two-sided binomial sign test on discordant pairs
    n = b + c; k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
for name, f in [("raw", raw), ("strict", strict)]:
    G = {i: eq(f(g[i]["output"]), g[i]["gold"]) for i in g}
    res = {}
    for k in (1, 3, 5):
        res[k] = {i: eq(vote([f(t) for t in s[i]["output"][:k]]), s[i]["gold"]) for i in s}
    orc = {i: any(eq(f(t), s[i]["gold"]) for t in s[i]["output"]) for i in s}
    print(f"[{name} scorer] greedy={sum(G.values())}", " ".join(f"maj@{k}={sum(v.values())}" for k, v in res.items()), f"pass@5={sum(orc.values())}")
    for k in (1, 5):
        b = sum(1 for i in G if not G[i] and res[k][i]); c = sum(1 for i in G if G[i] and not res[k][i])
        print(f"   maj@{k} vs greedy: fixed={b} broken={c} exact sign-test p={mcnemar(b, c):.3f}")
    # first-sample maj@1 averaged over all 5 samples (less noisy estimate of single T=0.7 sample)
    avg1 = sum(eq(f(t), s[i]["gold"]) for i in s for t in s[i]["output"]) / 5
    print(f"   mean single T=0.7 sample acc={avg1:.1f}")
    if name == "raw":
        print("   check: maj@5 raw matches logged ok:", sum(r["ok"] for r in s.values()) == sum(res[5].values()))
        disagree = [i for i in g if raw(g[i]["output"]) != strict(g[i]["output"])]
print("greedy outputs where scorers disagree:", len(disagree))
for i in disagree[:8]:
    print(i, "gold", g[i]["gold"], "raw", raw(g[i]["output"]), "strict", strict(g[i]["output"]), "| tail:", g[i]["output"][-120:].replace("\n", " "))
print("tokens/q greedy=%.0f sc=%.0f" % (sum(r["tokens"] for r in g.values())/100, sum(r["tokens"] for r in s.values())/100))
