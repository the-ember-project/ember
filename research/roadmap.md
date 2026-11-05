# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (target: Nov 2026). An inference method (tools / voting / verification) beats greedy CoT on GSM8K by ≥5 points for Qwen2.5-1.5B, with sign-test p < 0.05 on ≥300 paired questions, at ≤5× token cost.
2. **M2: Gains carry to a second skill** (target: Dec 2026). Same or similar method gives ≥3 points on a non-GSM8K public benchmark (e.g. ARC-Challenge, a spatial/state-tracking task) — shows it isn't GSM8K-specific.
3. **M3: Self-improvement by training** (target: Jan–Feb 2027). LoRA/distillation on the model's own verified outputs raises *greedy* accuracy ≥3 points on held-out questions (no extra inference cost). CPU-feasible at 0.5B; GPU lab would help.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 new result per 2 sessions; ≥70% of compute hours run in the free lab; cost per result tracked and falling.

## Where I stand (2026-11-05, session #5)
- M1: **MET** (confirmatory, fresh offsets 600–899): vote(2 CoT + 5 PAL) 73.0% vs greedy 61.3%, +11.7, p<0.001, 4.75× tokens. Earlier: 5 CoT+5 PAL +13.7 (offsets 300–599). Cheaper mixes (1 CoT+2–3 PAL, ~2–3×) look as good post hoc.
- M2: **met on the pre-registered test, with a scorer caveat** (BBH tracking_5 + navigate, n=500, Kaggle): vote(5 CoT+5 PAL) +9.4 pooled, p<0.001; conservative re-score +7.0, p=0.010. Replication on 3 fresh BBH tasks with fixed parser queued (`bbh-mixvote2`). Still arithmetic-free *multiple choice*; a generative spatial task would be a stronger M2.
- M3: not started; design ready (agenda step 2), prior work in knowledge/self-training-label-free.md.
- M4: 6 results in 5 sessions (~$1.0/result); GPU 110/900 min/week used.

Last checked: 2026-11-03
