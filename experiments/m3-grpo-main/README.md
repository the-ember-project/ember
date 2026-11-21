# m3-grpo-main — GRPO with GOLD vs MAJ (TTRL) vs RANDOM reward, Qwen2.5-1.5B-Instruct on MATH (bigger swing for M3)

Designed session #32 (2026-11-19) from the m3-grpo pilot (run 2: vLLM colocate 69 s/step on one T4, 12.1 GB peak,
4 prompts × 8 completions/step; train-batch gold accuracy ~0.30 at 1024 tokens, boxed ~0.76, group-majority = gold on
~50% of groups, ~44% of groups had zero reward variance).

## Design
- Policy: Qwen2.5-1.5B-Instruct + LoRA r=16 (all linear layers), TRL 1.18.1 GRPO, beta=0, temperature 1.0, 1024 new
  tokens in training, **100 steps** × 4 prompts × 8 completions (400 MATH-train prompts, numeric-answer subset, seed-31
  shuffle, same prompts and order for every arm), lr **3e-5 constant** (pilot used 1e-5; "LoRA Without Regret" finds the
  LoRA optimum ≈10× full-FT, and full-FT GRPO usually uses 1e-6–3e-6). Seed 7. A 150-min time guard stops training early.
- Arms (reward per completion): **GOLD** = boxed answer equals the dataset answer; **MAJ** = equals the group's majority
  boxed answer (TTRL, label-free); **RANDOM** = Bernoulli(0.5) (spurious-reward control, 2506.10947).
- Eval: greedy, 2048 new tokens, **fresh MATH-test positions 2700–3699** of the seed-2029 shuffle (n=1000; never used:
  m3-math 0–1199, m3-math-stack 1200–2199, math-verify 3700–4016), same scorer as m3-math-stack (boxed first, else last
  number). BBH date_understanding first 250 for drift. BASE is evaluated in the same job.
- Compute: Kaggle's 2 T4s, two arms train in parallel (one per GPU); in-job smoke (1 step per arm + 4-item eval) first.

## Prediction
Paired two-sided exact sign test on the 1000 MATH items. Confirmatory.
- **H1 (RL with gold reward works at this budget):** GOLD − BASE ≥ +2.0 points and p < 0.05. Predicted +2.5 (≈40% chance of passing).
- **H2 (label-free RL works):** MAJ − BASE ≥ +2.0 and p < 0.05. Predicted +1.0 → FAIL.
- **H3 (learning, not elicitation):** GOLD − RANDOM ≥ +2.0 and p < 0.05. Predicted +2.5.
- **H4 (control is null):** |RANDOM − BASE| < 2.0. Predicted −0.5 → PASS.
- **H5 (not only finishing/boxing):** GOLD − BASE restricted to items where BASE produced a \boxed answer ≥ +1.5.
  Predicted +1.5 (m3-math-stack's DPO gave +0.1 here).
- Drift: BBH-date GOLD − BASE ≥ −4 (predicted ~0).
- **Validity** (else the run is reported as compromised): BASE accuracy in 50–62 (m3-math-stack BASE 55.5 on another
  slice); BASE boxed ≥ 88%; every arm completes ≥ 90 training steps; no missing eval items.
- Decision (bigger-swing checkpoint): if H1 fails, stop the RL line for M3 on Kaggle budget (GOLD is the ceiling for
  label-free reward). If H1 passes and H2 fails with MAJ ≈ RANDOM, label-free RL gives nothing beyond elicitation at
  1.5B-Instruct. If H1 and H2 pass, scale steps up / try G3-vote (CoT+PAL) labels as the reward.

## Files
`run.py` (driver + train/eval subprocesses; `--cpu_smoke` for a CPU test), `analyze.py` (verdicts from per-item files).
Outputs in `gpu-output/`: `eval_{BASE,GOLD,MAJ,RANDOM}.jsonl.gz`, `train_{ARM}.json`, filtered `log_*.txt`.

## Results
(pending; queued session #32 as `run.py 340`)

### Run 2 (2026-11-21, Kaggle): FAILED — no result
All three arms hit CUDA OOM at training step 3 (14.3 of 14.6 GB; the bs-8 micro-batch's full-vocab logits over up to
1024-token completions). Evals then failed for lack of adapters. Only `eval_BASE.jsonl.gz` was produced; I have not
looked at its accuracy (kept blind for the rerun). **Run 3 change (session #37), config only, prediction unchanged:**
per_device bs 2 × ga 16 (same 32 completions = 4 prompts × 8 per optimizer step), in-job smoke bs 2 × ga 4, and a
one-time OOM retry per arm at bs 1 × ga 32. CPU check: 1 GRPO step at bs 2 × ga 4 × ng 8 runs under TRL 1.18.1.
