"""Verdicts for m3-stack from per-question files (gpu-output/) + stored 3B greedy (m3-vs-3b)."""
import gzip, json, math, os
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, "gpu-output")
def sign_p(f, b):
    n = f + b; k = min(f, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def ev(path):
    return {json.loads(l)["idx"]: json.loads(l)["correct"] for l in open(path) if '"idx"' in l}
acc = {"3B": ev(os.path.join(H, "..", "m3-vs-3b", "gpu-output", "eval_BASE.jsonl"))}
TRAIN = os.path.exists(os.path.join(G, "g3_MIXR.jsonl.gz"))  # #30: train-split steps now run last and may be missing
def cmp(x, y, tag):
    f = sum(acc[x][i] and not acc[y][i] for i in ids); b = sum(acc[y][i] and not acc[x][i] for i in ids)
    print(f"{tag}: {x}-{y} {100*(f-b)/n:+.1f} (fixed {f} / broken {b}) p={sign_p(f, b):.4g}")
if TRAIN:
    for arm in ("BASE", "MIXR"):
        acc[arm] = ev(os.path.join(G, f"eval_{arm}.jsonl"))
        acc[f"G3_{arm}"] = {json.loads(l)["idx"]: json.loads(l)["correct"] for l in gzip.open(os.path.join(G, f"g3_{arm}.jsonl.gz"), "rt")}
    ids = sorted(set.intersection(*[set(v) for v in acc.values()])); n = len(ids)
    print("n paired", n, {k: round(100 * sum(v[i] for i in ids) / n, 1) for k, v in acc.items()})
    cmp("G3_MIXR", "3B", "H1 (pass if >= -2.0)")
    cmp("G3_MIXR", "MIXR", "H2 (pass if >= +2.0, p<.05)")
    cmp("G3_MIXR", "G3_BASE", "H3 (pass if >= +2.0, p<.05)")
    cmp("G3_BASE", "BASE", "expl")
    cmp("MIXR", "BASE", "expl")
    cmp("G3_BASE", "3B", "expl")
    s = json.load(open(os.path.join(G, "summary.json"))); g = json.load(open(os.path.join(G, "g3_summary.json")))
    print("greedy tok/q", s.get("greedy_tok_per_q"), "g3", g, "bbh", s.get("bbh_date_acc"))
    t3 = json.load(open(os.path.join(H, "..", "m3-vs-3b", "gpu-output", "summary.json")))
    print("3B summary", t3)

# ---- session #24: GSM8K TEST split (full 1319) — H4–H6 ----
if os.path.exists(os.path.join(G, "test_g3_MIXR.jsonl.gz")):
    acc = {"3B": ev(os.path.join(G, "test3b_eval_BASE.jsonl"))}
    for arm in ("BASE", "MIXR"):
        acc[arm] = ev(os.path.join(G, f"test_eval_{arm}.jsonl"))
        acc[f"G3_{arm}"] = {json.loads(l)["idx"]: json.loads(l)["correct"] for l in gzip.open(os.path.join(G, f"test_g3_{arm}.jsonl.gz"), "rt")}
    ids = sorted(set.intersection(*[set(v) for v in acc.values()])); n = len(ids)
    print("\nTEST n paired", n, {k: round(100 * sum(v[i] for i in ids) / n, 1) for k, v in acc.items()})
    cmp("MIXR", "BASE", "H4 test (pass if >= +2.0, p<.05)")
    cmp("G3_MIXR", "G3_BASE", "H5 test (pass if >= +2.0, p<.05)")
    cmp("G3_MIXR", "3B", "H6 test (pass if >= -2.0)")
    cmp("G3_MIXR", "MIXR", "expl test")
    cmp("G3_BASE", "BASE", "expl test")
    cmp("3B", "BASE", "expl test")
    print("test g3", json.load(open(os.path.join(G, "test_g3_summary.json"))),
          "test greedy tok/q", json.load(open(os.path.join(G, "test_summary.json"))).get("greedy_tok_per_q"),
          "3B tok/q", json.load(open(os.path.join(G, "test3b_summary.json"))).get("greedy_tok_per_q"))
