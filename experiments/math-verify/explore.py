# Exploratory (session #35): within-question AUROC of the ORM and rescue of lone-correct items, for comparison
# with math-selfverify (self-judge within-q 0.677) and math-select (30/37 rescuable misses had one correct sample).
import sys; sys.argv=[sys.argv[0]]
exec(open("analyze.py").read().split("n = len(items)")[0].replace('print(', '(lambda *a, **k: None)('))
for size in ("1.5B", "3B"):
    g = G[size]; S = scores(size, "rm17")["scores"]
    num = den = 0; resc = lone = lone_ok = 0; miss = 0
    for i in items:
        c = [ok(v, g[i]["gold"]) for v in g[i]["ans"]]; s = S[str(i)]
        P = [x for x, y in zip(s, c) if y]; N = [x for x, y in zip(s, c) if not y]
        if P and N:
            a = auroc(P, N); num += a; den += 1
        sc = acc[f"SC8_{size}"][i]
        if any(c) and not sc:
            miss += 1; resc += acc[f"WSC8_{size}_rm17"][i]
            if sum(c) == 1: lone += 1; lone_ok += acc[f"BoN8_{size}_rm17"][i]
    print(f"{size}: within-q AUROC (mean over {den} mixed items) {num/den:.3f}; SC8 misses with a correct sample {miss}, "
          f"WSC8 rescues {resc}; lone-correct {lone}, BoN picks it {lone_ok}")

# N-curve: WSC-k / BoN-k / SC-k using the first k samples (greedy first), both sizes.
print("k:   " + " ".join(f"{k:>5}" for k in range(1, 9)))
for size in ("1.5B", "3B"):
    g = G[size]; S = scores(size, "rm17")["scores"]
    for name, f in (("SC", lambda i, k: wvote(g[i]["ans"][:k], [1]*k)),
                    ("WSC", lambda i, k: wvote(g[i]["ans"][:k], [sig(x) for x in S[str(i)][:k]])),
                    ("BoN", lambda i, k: g[i]["ans"][max(range(k), key=lambda j: S[str(i)][j])]),
                    ("pass", None)):
        row = []
        for k in range(1, 9):
            if f is None: c = sum(any(ok(v, g[i]["gold"]) for v in g[i]["ans"][:k]) for i in items)
            else: c = sum(ok(f(i, k), g[i]["gold"]) for i in items)
            row.append(100 * c / len(items))
        print(f"{size} {name:4} " + " ".join(f"{x:5.1f}" for x in row))
