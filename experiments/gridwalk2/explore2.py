"""Exploratory (session #14, after n=200 verdicts): is the gain from mixing or from the program format?
Paired comparisons among arms, same vote/sign-test as analyze.py. Also: 5P with PAL-first vs CoT-first tie-breaks."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import analyze as A
ok = {}
for name, f in A.arms.items():
    ok[name] = {i: A.vote(f(i)[0]) == A.G[i]["gold"] for i in A.ids}
ok["PAL1"] = {i: A.P[i]["preds"][0] == A.G[i]["gold"] for i in A.ids}
ok["vote(5P)+greedy"] = {i: A.vote(A.P[i]["preds"][:5] + [A.G[i]["preds"][0]]) == A.G[i]["gold"] for i in A.ids}
ok["vote(3P)"] = {i: A.vote(A.P[i]["preds"][:3]) == A.G[i]["gold"] for i in A.ids}
def cmp(a, b):
    fx = sum(ok[a][i] and not ok[b][i] for i in A.ids); br = sum(ok[b][i] and not ok[a][i] for i in A.ids)
    print(f"{a:28s} - {b:28s} = {100*(fx-br)/len(A.ids):+.1f} ({fx}/{br}) p={A.signp(fx, br):.4f}")
for a, b in [("vote(5P)", "MIX10 vote(5C+5P) [H1]"), ("vote(5P)", "G3 vote(greedy+3P) [H3]"),
             ("vote(5P)", "PAL1"), ("vote(3P)", "PAL1"), ("G3 vote(greedy+3P) [H3]", "vote(3P)"),
             ("vote(2C+3P)", "vote(5P)"), ("vote(5P)+greedy", "vote(5P)"), ("PAL1", "greedy")]:
    cmp(a, b)
