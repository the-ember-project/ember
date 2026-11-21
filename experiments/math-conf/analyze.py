"""Verdicts for math-conf (rules fixed in README before the run). Usage: python analyze.py [conf.jsonl] [math-verify outdir]"""
import gzip, json, math, os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
CF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "conf_1.5B.jsonl")
D = sys.argv[2] if len(sys.argv) > 2 else os.path.join(H, "..", "math-verify", "gpu-output", "out")
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def ok(v, g): return v is not None and abs(v - g) <= 1e-4 * max(1.0, abs(g))
def wvote(ans, w):  # weighted plurality, ties to the earliest (same rule as math-verify)
    s, first = defaultdict(float), {}
    for j, (v, x) in enumerate(zip(ans, w)):
        if key(v) is None or x is None: continue
        k = key(v); s[k] += x; first.setdefault(k, j)
    return max(s, key=lambda k: (s[k], -first[k])) if s else None
def sign_p(f, b):
    n = f + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
def sig(x): return 1 / (1 + math.exp(-x))
G = {r["idx"]: r for r in (json.loads(l) for l in gzip.open(os.path.join(D, "gen_1.5B.jsonl.gz"), "rt"))}
RMp = os.path.join(D, "rm_1.5B_rm17.json"); RM = json.load(open(RMp))["scores"] if os.path.exists(RMp) else None
C = {r["idx"]: r for r in (json.loads(l) for l in open(CF) if l.strip())}
mixed = [i for i, r in G.items() if len({key(x) for x in r["ans"]} - {None}) >= 2]
nf = sum(len(C[i]["feats"]) for i in mixed if i in C); ok_f = sum(f is not None for i in mixed if i in C for f in C[i]["feats"])
al = [abs(f["n"] - t) / max(1, t) <= 0.15 for i in mixed if i in C for f, t in zip(C[i]["feats"], G[i]["toks"]) if f]
print(f"items {len(G)}, mixed {len(mixed)}, scored {sum(i in C for i in mixed)}; samples finite {ok_f}/{8*len(mixed)} "
      f"({100*ok_f/max(1,8*len(mixed)):.1f}%, need >=95); length-aligned {100*sum(al)/max(1,len(al)):.1f}% (need >=90)")
acc = defaultdict(dict); wq = defaultdict(list)
for i, r in G.items():
    ans, g = r["ans"], r["gold"]; acc["SC8"][i] = ok(wvote(ans, [1] * len(ans)), g)
    acc["pass@8"][i] = any(ok(a, g) for a in ans)
    if RM is not None: acc["WSC8_rm17"][i] = ok(wvote(ans, [sig(x) for x in RM[str(i)]]), g)
    F = C[i]["feats"] if i in C else None
    if F is None or i not in mixed:  # unanimous (or unscored): every selector returns the vote
        for k in ("DCF", "CW8", "SCB", "BoN_conf"): acc[k][i] = acc["SC8"][i]
        continue
    b = [f["bottom10"] if f else None for f in F]
    vals = sorted([x for x in b if x is not None], reverse=True); thr = vals[len(vals) // 2 - 1] if len(vals) >= 2 else -1e18
    acc["DCF"][i] = ok(wvote(ans, [x if x is not None and x >= thr else None for x in b]), g)
    acc["CW8"][i] = ok(wvote(ans, b), g)
    sc = [f["self_cert"] if f else -1e18 for f in F]; rk = sorted(range(len(sc)), key=lambda j: -sc[j])
    acc["SCB"][i] = ok(wvote(ans, [(len(sc) - rk.index(j)) ** 0.5 for j in range(len(sc))]), g)
    acc["BoN_conf"][i] = ok(ans[max(range(len(b)), key=lambda j: (b[j] if b[j] is not None else -1e18, -j))], g)
    cor = [ok(a, g) for a in ans]
    for name in ("bottom10", "mean_conf", "lowest", "tail", "self_cert", "mean_lp"):
        xs = [f[name] if f else None for f in F]
        p = [x for x, c in zip(xs, cor) if c and x is not None]; q = [x for x, c in zip(xs, cor) if not c and x is not None]
        if p and q: wq[name].append(auroc(p, q))
n = len(G)
for k in acc: print(f"{k:10s} {100*sum(acc[k].values())/n:5.1f}")
def cmp(x, y, lab):
    f = sum(acc[x][i] and not acc[y][i] for i in G); b = sum(acc[y][i] and not acc[x][i] for i in G)
    print(f"{lab}: {x} - {y} = {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign_p(f, b):.4g}")
cmp("DCF", "SC8", "H1 (>= +2, p<.05)")
if RM is not None: cmp("DCF", "WSC8_rm17", "H2 (>= -1)")
for x in ("CW8", "SCB", "BoN_conf"): cmp(x, "SC8", "expl")
for name, v in wq.items(): print(f"within-question AUROC {name:10s} {sum(v)/len(v):.3f} over {len(v)} questions" + ("  <- H3 (>= 0.60)" if name == "bottom10" else ""))
