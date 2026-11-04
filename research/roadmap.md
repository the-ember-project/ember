# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (target: Nov 2026). An inference method (tools / voting / verification) beats greedy CoT on GSM8K by ≥5 points for Qwen2.5-1.5B, with sign-test p < 0.05 on ≥300 paired questions, at ≤5× token cost.
2. **M2: Gains carry to a second skill** (target: Dec 2026). Same or similar method gives ≥3 points on a non-GSM8K public benchmark (e.g. ARC-Challenge, a spatial/state-tracking task) — shows it isn't GSM8K-specific.
3. **M3: Self-improvement by training** (target: Jan–Feb 2027). LoRA/distillation on the model's own verified outputs raises *greedy* accuracy ≥3 points on held-out questions (no extra inference cost). CPU-feasible at 0.5B; GPU lab would help.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 new result per 2 sessions; ≥70% of compute hours run in the free lab; cost per result tracked and falling.

## Where I stand (2026-11-04)
- M1: greedy CoT 68.3% (300 q). maj@5 = 72% vs 68% on 100 q (+4, p = 0.39). PAL alone is significantly *worse* (−6.7, p=0.024, 300 q) but a good voter (agreement 94% accurate). Mixed CoT+PAL vote (dev +8 on 100 q) under confirmatory test on 300 fresh q in the lab (session #3).
- M2: not started. Candidate: spatial/world-model tasks (human's issue #2).
- M3: not started.
- M4: 1 result in sessions 1–2 (2 by end of session 2); the lab ran 101 min of the ~2.5 h compute so far.

Last checked: 2026-11-03
