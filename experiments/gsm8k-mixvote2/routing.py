"""EXPLORATORY: greedy-anchored cheap mixes (G3 etc., found post hoc on bbh-mixvote2) checked on GSM8K offsets 600-899."""
import sys; sys.argv = sys.argv[:1]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import g, c2, pk, eq, vote, sign, ids, gt, ct, pt
gold = {i: g[i]["gold"] for i in ids}
R = {"greedy": (lambda i: g[i]["pred"], 1.0),
     "G3 vote(greedyCoT+3PAL)": (lambda i: vote([g[i]["pred"]] + pk[i]["preds"][:3]), 1 + 3*pt/gt),
     "G2 vote(greedyCoT+2PAL)": (lambda i: vote([g[i]["pred"]] + pk[i]["preds"][:2]), 1 + 2*pt/gt),
     "G4 vote(greedyCoT+4PAL)": (lambda i: vote([g[i]["pred"]] + pk[i]["preds"][:4]), 1 + 4*pt/gt),
     "ref vote(1CoTsample+3PAL)": (lambda i: vote(c2[i]["preds"][:1] + pk[i]["preds"][:3]), (ct+3*pt)/gt),
     "ref P2 vote(1CoTsample+4PAL)": (lambda i: vote(c2[i]["preds"][:1] + pk[i]["preds"][:4]), (ct+4*pt)/gt)}
ok = {k: {i: eq(f(i), gold[i]) for i in ids} for k, (f, _) in R.items()}; n = len(ids)
for k, (_, c) in R.items():
    b = sum(ok[k][i] and not ok["greedy"][i] for i in ids); d = sum(ok["greedy"][i] and not ok[k][i] for i in ids)
    print(f"{k:30s} acc={sum(ok[k].values())/n:.3f} diff={100*(b-d)/n:+.1f} fixed={b} broken={d} p={sign(b,d):.3f} cost={c:.2f}x")
