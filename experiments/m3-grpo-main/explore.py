"""Exploratory extras for m3-grpo-main (run 3): GOLD vs MAJ, by level, near-cap lengths."""
import sys, collections
from analyze import load, paired
E = {a: load(a)[0] for a in ("BASE", "GOLD", "MAJ", "RANDOM")}
print("GOLD-MAJ", paired(E["GOLD"], E["MAJ"]))
print("MAJ+GOLD union vs BASE (either correct):", sum(g["correct"] or m["correct"] for g, m in zip(E["GOLD"], E["MAJ"])) / 10)
for a, m in E.items():
    lv = collections.defaultdict(list)
    for r in m: lv[r["level"]].append(r["correct"])
    cap = sum(not r["boxed"] for r in m)
    print(a, "unboxed:", cap, " ".join(f"{k[-1]}:{100*sum(v)/len(v):.0f}" for k, v in sorted(lv.items())))
for a in ("GOLD", "MAJ", "RANDOM"):
    for L in ("Level 1", "Level 2", "Level 3", "Level 4", "Level 5"):
        mask = [r["level"] == L for r in E["BASE"]]
        r = paired(E[a], E["BASE"], mask); print(a, L, r["diff"], r["fixed"], r["broken"], round(r["p"], 3))
