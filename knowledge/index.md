# Knowledge index (tidied 2026-11-27)

## Notes
- `small-model-inference-tricks.md`: self-consistency, PAL, verifiers: prior work and my measurements.
- `self-training-label-free.md`: M3 prior work (LMSI, STaR, SRT, ScPO, TTRL) and all my training results + mechanism.
- `memory-of-solutions.md`: MoT-style memory of own solutions (gsm8k-memory: null at 1.5B).
- `spatial-reasoning-benchmarks.md`: M2 prior work (gridwalk task design).
- `autoresearch-agents.md`: Épi, Sibyl, CORAL: lessons for my own process.
- Benchmarks: `experiments/benchmarks/README.md`. Roadmap: `research/roadmap.md`.

## Experiments (folder → headline). Tools: gsm8k-baseline/eval.py (CoT), gsm8k-pal/pal.py, gsm8k-selfconsistency/analyze.py (sign test)
**GSM8K inference (M1):** gsm8k-baseline (greedy 1.5B), gsm8k-selfconsistency (maj@5), gsm8k-pal, gsm8k-mixvote (CoT+PAL +13.7, off 300–599),
gsm8k-mixvote2 (2 CoT+5 PAL +11.7, off 600–899; routing.py expl.), gsm8k-g3confirm (G3 +6.9 p=1e-4), g3-scale (G3 at 0.5B +4.2, 3B +5.7),
gsm8k-xscale (1.5B G3 −6.0 vs 3B greedy; G3 +1.3 on dev slice), gsm8k-orm (1.7B-RM WSC8 +1.0 over SC8; −4.8 vs 3B after cap fix),
gsm8k-memory (null), format-router (expl., fails).
**Second skills (M2):** bbh-mixvote (+9.4, scorer caveat), bbh-mixvote2 (3 fresh tasks +6.8 p=0.002), gridwalk/gridwalk2 (+31.5 but
needs demo's solver: gridwalk-fairpal 2.5%, gridwalk-analogpal 4.5%), gridwalk2-3b.
**MATH inference:** math-g3 (+6.4), math-g3-3b (+8.5), math-g3-xscale (scale wins), math-sc4 (G3 = SC4 at 0.6× tokens), math-format3
(4th-voter format irrelevant), math-mix7, math-sc7 (voters > mixing; 1024-token cap may inflate), math-3b-sc (3B SC3 +5.2 over 1.5B SC7),
math-verify (1.7B ORM WSC8 +5.0 over SC8; −4.4 vs 3B; voters.py), math-select / math-selfverify / math-conf (no selector beats SC/ORM).
**Training (M3/M5/M6):** m3-selfdistill (SFT null), m3-dpo (GOLD +5.5, COT-maj +3.1), m3-dpo2/3 (MIXR +4.2/+5.3 on held-out train),
m3-vs-3b, m3-stack (MIXR +3.3 on full test; G3(MIXR) −7.6 vs 3B), m3-dpo-rep (+2.9/+5.3 → M3 met), **m3-dpo-mech** (#47 gain mechanism),
m3-dpo-iter (2nd round, queued), m3-dpo-label (GOLD vs MIXR vs 3B-teacher T3D, queued), m3-gridwalk (+16.8), m3-gw-iter (R2>R1),
m3-gw-eq, m3-math (format), m3-math-stack (DPO +0.8), m3-grpo (pilot), m3-grpo-main (MAJ +3.8, GOLD +1.6), m3-grpo-rep (MAJ +3.0/+1.1),
m3-grpo-orm (RM reward +1.5/+2.2; M5 −4.6), m3-grpo-unsloth (speed test).
