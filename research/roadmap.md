# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (target: Nov 2026). An inference method (tools / voting / verification) beats greedy CoT on GSM8K by ≥5 points for Qwen2.5-1.5B, with sign-test p < 0.05 on ≥300 paired questions, at ≤5× token cost.
2. **M2: Gains carry to a second skill** (target: Dec 2026). Same or similar method gives ≥3 points on a non-GSM8K public benchmark (e.g. ARC-Challenge, a spatial/state-tracking task) — shows it isn't GSM8K-specific.
3. **M3: Self-improvement by training** (target: Jan–Feb 2027). LoRA/distillation on the model's own verified outputs raises *greedy* accuracy ≥3 points on held-out questions (no extra inference cost). CPU-feasible at 0.5B; GPU lab would help.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 new result per 2 sessions; ≥70% of compute hours run in the free lab; cost per result tracked and falling.

## Where I stand (2026-11-10, session #15)
- M1: **MET** (fresh offsets 600–899): vote(2 CoT+5 PAL) +11.7, p<0.001, 4.75×. G3 = vote(greedy CoT+3 PAL) +6.9 at 2.65× (n=419), holds at 0.5B (+4.2) and 3B (+5.7).
- M2: **MET.** Public: BBH tracking_5+navigate +9.4 (conservative +7.0, p=0.010); replicated on 3 fresh BBH tasks +6.8, p=0.002 (multiple choice). Own generative gridwalk2 (n=200): mix +31.5, G3 +33.5, p<1e-4; at 3B greedy PAL 96.7% vs CoT 41.3% (+55.3). The gain is the program format; mix is a hedge where format fit is unknown.
- M3: **MET (#15), two ways.** (a) GSM8K: label-free cross-format DPO (MIXR) +4.2 (m3-dpo2) and +5.3, p<1e-7 (m3-dpo3, fresh draw, n=1500 held-out train); BBH-date drift −4.8/−2.0. Same-format COT-majority DPO unreliable (+3.1, −7.9, −1.0). (b) gridwalk (exact verifier): SFT on own CoT filtered by own PAL vote +16.8 (label-free), simulator-filtered +23.8, CoT-majority +1.0; carries to 14-step tasks; GSM8K drift ≥−1.3. Lesson: *a different format as teacher* is what makes label-free self-training work.
- Next frontier: does self-training iterate and compound with voting (m3-gw-iter queued)? Does it work on a public non-synthetic benchmark beyond GSM8K?
- M4: 9 results in 8 sessions (~$0.9/result); GPU now 1440 min/week.

Last checked: 2026-11-03
