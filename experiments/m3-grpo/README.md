# m3-grpo — RL (GRPO) instead of DPO for M3 (bigger swing, started session #30)

Exploratory: pilot.py is a speed/feasibility pilot (no hypothesis tested). It runs TRL 1.18.1 GRPO + LoRA (r=16) on
Qwen2.5-1.5B-Instruct, MATH-train numeric subset, GOLD reward, 4 prompts x 8 completions per optimizer step, 1024 new
tokens, first with vLLM colocate (12 steps), then plain HF generation (2 steps, 15-min cap). It records s/step, reward, completion
length / truncation, GPU memory, and TTRL label precision (how often the group-majority answer equals gold).

## Why
Label-free DPO did not give M3 on uncontaminated data (MATH +0.8 at 2048 tokens, m3-math-stack; GSM8K test MIX +2.35
p=0.07). Published label-free RL gains (TTRL, Spurious Rewards 2506.10947) are mostly on Qwen2.5-*Math* base models,
where even random rewards help (format elicitation). Open question for an *instruct* 1.5B model that already formats
well: does on-policy RL with a majority-vote reward improve greedy reasoning, and does a random-reward control get the
same gain (= elicitation, not learning)? Prior art: TTRL 2504.16084; GRPO on Qwen2.5-1.5B-Instruct on one Kaggle T4
has been done with Unsloth+TRL (HF model raju-ai/qwen2.5-1.5b-grpo-gsm8k), so feasibility is plausible.

## Plan (main experiment designed after the pilot; its Prediction will be written before it runs)
Arms: GOLD vs MAJ (TTRL: reward = agreement with the group's majority answer) vs RANDOM (p=0.5), same prompts/seed/steps.
Eval: greedy at 2048 tokens on a fresh MATH-test slice (positions not used by m3-math / m3-math-stack), n≥1000, paired
sign tests, plus BBH-date drift. Sketch of the predictions: GOLD − BASE ≥ +3; MAJ ≥ half of GOLD's gain; RANDOM ≈ 0.

## Decision rule (bigger-swing checkpoint: session #33)
Stop the RL line if the pilot shows < 25 optimizer steps/hour (i.e. < 100 steps in 4 h of T4) for the faster config,
or if GOLD itself gives < +2 over BASE in the main run (then label-free RL can't be expected to beat it).

## Pilot results
- **Run 1 (2026-11-18, 8 GPU min): FAILED before any step.** Both configs died building the LoRA model: Kaggle's torchao
  0.10.0 is below peft 0.21.2's minimum (0.16) and peft raises instead of ignoring it (same clash as m3 run 1). Installed
  versions after `pip install trl[vllm]==1.18.1`: torch 2.14.0, vllm 0.33.0, transformers 5.20.2, peft 0.21.2.
  Fix (#31): `run_one` patches `is_torchao_available` → False in the three peft modules that call it. CPU smoke with
  torchao 0.10.0 installed now passes for GOLD/MAJ/RANDOM. Run 1's log is kept in `gpu-output/` until run 2 overwrites it.
- **Run 2 (2026-11-19, 30 GPU min): worked.** vLLM colocate: median **69 s/step (≈52 steps/h)**, 12 steps, peak GPU
  memory 12.1 GB; HF generation: 130–204 s/step (2 steps). Train-batch stats (vLLM, 12 steps): gold accuracy 0.03–0.69
  (mean ≈0.30) at 1024 tokens, boxed 0.62–0.97 (mean ≈0.77), clipped 0–22%, group-majority = gold on ≈50% of groups,
  frac of zero-variance groups ≈0.44. Speed passes the ≥25 steps/h rule → main experiment designed in
  `experiments/m3-grpo-main/` (session #32).
