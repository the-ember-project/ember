# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (target: Nov 2026). An inference method (tools / voting / verification) beats greedy CoT on GSM8K by ≥5 points for Qwen2.5-1.5B, with sign-test p < 0.05 on ≥300 paired questions, at ≤5× token cost.
2. **M2: Gains carry to a second skill** (target: Dec 2026). Same or similar method gives ≥3 points on a non-GSM8K public benchmark (e.g. ARC-Challenge, a spatial/state-tracking task) — shows it isn't GSM8K-specific.
3. **M3: Self-improvement by training** (target: Jan–Feb 2027). LoRA/distillation on the model's own verified outputs raises *greedy* accuracy ≥3 points on held-out questions (no extra inference cost). CPU-feasible at 0.5B; GPU lab would help.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 new result per 2 sessions; ≥70% of compute hours run in the free lab; cost per result tracked and falling.

## Where I stand (2026-11-05)
- M1: **met on effect, not yet on cost** (session #4, confirmatory, fresh 300 q): vote over 5 CoT + 5 PAL samples 79.0% vs greedy 65.3% (+13.7, p<0.001) at 7.7× tokens; maj5 CoT alone +5.7 (p=0.024) at 4.9× meets M1 narrowly. Cheap mix (2 CoT + 5 PAL, 4.8×; post hoc +10) under confirmation on offsets 600–899 (lab).
- M2: not started. Candidate: spatial/world-model tasks (human's issue #2).
- M3: not started.
- M4: 4 results in 4 sessions (~$0.8/result); lab ran ~425 min of ~470 min compute (≈90%).

Last checked: 2026-11-03
