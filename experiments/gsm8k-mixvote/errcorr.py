"""Exploratory: are CoT and PAL errors less correlated across formats than within? (offsets 300-599)
For each question and each pair of samples (within CoT, within PAL, cross CoT-PAL) report
P(both wrong) and P(both wrong with the SAME wrong answer) -- the latter is what breaks votes."""
import json, os, itertools
D = os.path.dirname(os.path.abspath(__file__))
L = lambda f: {r["idx"]: r for r in map(json.loads, open(os.path.join(D, f)))}
ck, pk = L("cot_k5_off300_n300.jsonl"), L("pal_k5_off300_n300.jsonl")
eq = lambda a, b: a is not None and b is not None and abs(a - b) < 1e-4
st = {k: [0, 0, 0, 0] for k in ("CoT-CoT", "PAL-PAL", "CoT-PAL")}  # pairs, bothwrong, samewrong, acc-product baseline
for i in ck:
    gold = ck[i]["gold"]; C, P = ck[i]["preds"], pk[i]["preds"]
    for name, pairs in (("CoT-CoT", itertools.combinations(C, 2)), ("PAL-PAL", itertools.combinations(P, 2)), ("CoT-PAL", itertools.product(C, P))):
        for a, b in pairs:
            wa, wb = not eq(a, gold), not eq(b, gold); s = st[name]
            s[0] += 1; s[1] += wa and wb; s[2] += wa and wb and eq(a, b)
pc = 1 - sum(eq(v, ck[i]["gold"]) for i in ck for v in ck[i]["preds"]) / (5 * len(ck))
pp = 1 - sum(eq(v, pk[i]["gold"]) for i in pk for v in pk[i]["preds"]) / (5 * len(pk))
print(f"sample error rate: CoT {pc:.3f}, PAL {pp:.3f} (independence would give both-wrong CoT-CoT {pc*pc:.3f}, PAL-PAL {pp*pp:.3f}, CoT-PAL {pc*pp:.3f})")
for k, (n, bw, sw, _) in st.items():
    print(f"{k}: pairs={n} P(both wrong)={bw/n:.3f} P(same wrong answer)={sw/n:.3f} P(same | both wrong)={sw/max(bw,1):.3f}")
