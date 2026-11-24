# m3-grpo-orm — GRPO with a free reward model as the (label-free) reward, plus an M5 test

**Why:** m3-grpo-main run 3: majority-vote reward (MAJ) +3.8 greedy on MATH, gold reward only +1.6 — answer-label
precision (MAJ ~90% vs gold 100%) did not seem to be what limits GRPO here. A *dense* signal that ranks completions
*within* an answer group might be different in kind: the free Skywork-Reward-V2-Qwen3-0.6B ORM separates right from wrong
samples of this model (math-verify: WSC8 +4.1 over SC8 with the 0.6B; the 1.7B has within-question AUROC 0.81). No gold
labels are used (the ORM is an external general-purpose model; wording must say "label-free with an external verifier").
Second aim, **M5**: can 1.5B + this training + test-time ORM-weighted voting reach 3B greedy − 2 on the same items?

Prior work: RLHF-style RM rewards for math are standard (e.g. Math-Shepherd, PRM/ORM RL); RM rewards are known to be
hackable (length/style). TTRL/majority rewards: 2504.16084; Spurious Rewards 2506.10947; Lin 2605.19444 (sharpening).
What's new here: same-recipe, same-items comparison of MAJ vs a small off-the-shelf ORM reward at 1.5B instruct, ≥2 seeds.

**Design (recipe = m3-grpo-rep except the reward):** Qwen2.5-1.5B-Instruct, LoRA r16, GRPO (TRL 1.18.1, vLLM colocate,
gpu util 0.30 to make room for the RM), 100 steps × 4 prompts × 8 completions, 1024 tokens, lr 3e-5, beta 0.
Reward = raw Skywork-Reward-V2-Qwen3-0.6B score of [user: problem, assistant: completion] (same format as math-verify);
non-finite scores → batch minimum. Two replicates: ORM1 (seed 10, MATH-train shuffle 34), ORM2 (seed 11, shuffle 35).
Eval (same items as m3-grpo-rep): greedy, 2048 tokens, MATH test positions 0–999 (seed-2029 shuffle) for BASE, ORM1,
ORM2 and Qwen2.5-3B-Instruct (fp16 vLLM). Sampled: positions 0–499, greedy + 7 samples at T=0.7 (math-verify protocol)
for ORM1 and BASE, scored by the 0.6B RM; WSC8 = weighted plurality with weight sigmoid(RM score) per sample, ties to the earliest (same rule as
math-verify `analyze.py`).

## Prediction
Confirmatory. Paired two-sided exact sign tests. Written before any run of this code.
- **O1 (each replicate):** ORMk − BASE ≥ +2.0 and p < 0.05 (greedy, 1000 items). Predicted +1.5 each (~35% each passes;
  RM hacking risk).
- **O2 (M3 criterion):** mean(ORM1, ORM2) − BASE ≥ +3.0 and both p < 0.05. Predicted FAIL.
- **O3 (ORM vs MAJ reward):** mean(ORM1, ORM2) − mean(MAJ2, MAJ3) ≥ +1.5, using m3-grpo-rep's greedy outputs on the same
  1000 items (cross-job; also report ORM1 vs MAJ2 and ORM2 vs MAJ3 sign tests). Predicted −0.5 → FAIL. Untestable if
  m3-grpo-rep fails.
- **O4 (M5):** WSC8(ORM1) − 3B greedy ≥ −2.0 (point estimate) on positions 0–499. Predicted −4 → FAIL.
- **O5 (training adds to selection):** WSC8(ORM1) − WSC8(BASE) ≥ +1.5 on 0–499. Predicted +1.0 → FAIL.
- Descriptive (reward hacking): training-batch gold accuracy, boxed rate and RM score in steps 1–10 vs 91–100; mean
  eval response length ORM vs BASE.
- **Validity** (else compromised): BASE greedy 50–62 with ≥88% boxed; each ORM arm ≥90 steps; 3B greedy 60–75;
  non-finite RM scores after fallback 0; 1000 / 500 items per arm.
- Decision: O1 passes for both → the ORM reward works; compare with MAJ via O3. O4 PASS → M5 met on MATH.

## Files
- `run.py` (Kaggle driver; `--cpu_smoke` runs every phase on CPU with 0.5B), `orm_analyze.py` (verdicts).

## Status
- 2026-11-24 (#41): written; CPU smoke (`--cpu_smoke`: 0.5B, 1 GRPO step with the RM reward, eval incl. missing-adapter
  skip, sample, RM scoring) passes; `orm_analyze.py` tested on smoke outputs. Queued `run.py 250` behind m3-grpo-rep and
  m3-stack (est. ~205 GPU min: smoke 17, train ~120 in parallel, two eval chains ~62). Runs regardless of the
  m3-grpo-rep outcome: O1/O2 test a different reward and O4 is the M5 test. Memory risk: peak 12.6 GB at bs 2 with vLLM
  0.35 in m3-grpo-main; here vLLM 0.30 + 0.6B RM (~1.2 GB); OOM → one retry at bs 1, vLLM 0.27.
