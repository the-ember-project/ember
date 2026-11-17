"""Exploratory (written after H1/H2 seen): MIX7 vs SC5/SC6 at roughly matched tokens.
SCk = greedy + (k-1) of the 6 sampled CoT; averaged over all subsets to avoid picking one."""
import itertools, sys
sys.argv = ["x"]
exec(open("analyze.py").read().split("print(f\"n={n}")[0])
import statistics
for k in (5, 6):
    accs, diffs = [], []
    for sub in itertools.combinations(range(6), k - 1):
        a = []
        for c in C:
            r = B[c["idx"]]; s = r["cs"] + c["cs2"]
            a.append(ok(vote([r["cot"]] + [s[i] for i in sub]), r["gold"]))
        accs.append(100 * sum(a) / n)
    toks = sum(B[c["idx"]]["cot_tok"] + sum((B[c["idx"]]["cs_tok"] + c["cs2_tok"])) * (k - 1) / 6 for c in C) / sum(B[c["idx"]]["cot_tok"] for c in C)
    print(f"SC{k}: mean {statistics.mean(accs):.1f} (min {min(accs):.1f}, max {max(accs):.1f}) over {len(accs)} subsets, ~{toks:.2f}x tokens")
print(f"MIX7 {100*sum(A['MIX7'])/n:.1f}  SC7 {100*sum(A['SC7'])/n:.1f}" if A else "")
