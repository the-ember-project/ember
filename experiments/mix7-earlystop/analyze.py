"""Lossless sequential stopping for mix7 (gsm8k-mixvote2 data). python experiments/mix7-earlystop/analyze.py"""
import json, os
D = os.path.join(os.path.dirname(__file__), "..", "gsm8k-mixvote2")
def load(f): return {r["idx"]: r for r in map(json.loads, open(os.path.join(D, f)))}
c2, pk = load("cot_k2_off600_n300.jsonl"), load("pal_k5_off600_n300.jsonl")
def key(v): return None if v is None else round(v, 4)
def winner(votes):  # votes: list of (listpos, key) ; plurality, ties -> earliest list position
    s, first = {}, {}
    for p, k in votes:
        if k is None: continue
        s[k] = s.get(k, 0) + 1; first[k] = min(first.get(k, 99), p)
    return max(s, key=lambda k: (s[k], -first[k])) if s else None
def decided(votes, remaining):  # remaining: list of list positions not yet drawn
    r = len(remaining)
    if r == 0: return True
    w = winner(votes)
    s, first = {}, {}
    for p, k in votes:
        if k is None: continue
        s[k] = s.get(k, 0) + 1; first[k] = min(first.get(k, 99), p)
    if w is None: return False
    cw, fw = s[w], first[w]
    for k in s:  # existing rival gets all remaining votes
        if k == w: continue
        c, f = s[k] + r, min([first[k]] + remaining)
        if c > cw or (c == cw and f < fw): return False
    # brand-new answer gets all remaining votes
    c, f = r, min(remaining)
    if c > cw or (c == cw and f < fw): return False
    return True
ids = sorted(c2)
orders = {"a list order": [0, 1, 2, 3, 4, 5, 6], "b PAL first": [2, 3, 4, 5, 6, 0, 1]}
full_tok = sum(c2[i]["tokens"] + pk[i]["tokens"] for i in ids) / len(ids)
print(f"full mix7 tok/q {full_tok:.1f}")
res = {}
for name, order in orders.items():
    tok, same, ndraw, okc = 0, 0, 0, 0
    for i in ids:
        keys = [key(v) for v in c2[i]["preds"][:2]] + [key(v) for v in pk[i]["preds"][:5]]
        cost = [c2[i]["tokens"] / 2] * 2 + [pk[i]["tokens"] / 5] * 5
        full = winner(list(enumerate(keys)))
        drawn = []
        for j, p in enumerate(order):
            drawn.append((p, keys[p])); tok += cost[p]
            if decided(drawn, order[j + 1:]): break
        ndraw += len(drawn)
        w = winner(drawn); same += (w == full)
        okc += w is not None and abs(w - c2[i]["gold"]) < 1e-4
    t = tok / len(ids)
    res[name] = dict(tok_per_q=round(t, 1), saving_pct=round(100 * (1 - t / full_tok), 1), identical=same,
                     mean_voters=round(ndraw / len(ids), 2), acc=round(100 * okc / len(ids), 2))
    print(name, res[name])
print("E1", "PASS" if res["a list order"]["saving_pct"] >= 25 and res["a list order"]["identical"] == len(ids) else "FAIL")
print("E2", "PASS" if res["b PAL first"]["tok_per_q"] < res["a list order"]["tok_per_q"] and res["b PAL first"]["identical"] == len(ids) else "FAIL")
json.dump(dict(full_tok_per_q=round(full_tok, 1), **res), open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=1)
