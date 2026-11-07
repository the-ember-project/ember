# m3-dpo: label-free preference training, with the CoT+PAL vote picking the preferred rationale

Roadmap M3, run 3. In m3-selfdistill (run 2), SFT on the model's own rationales gave nothing, even with gold filtering (MIX 0.0, GOLD −0.5, n=419). Diagnosis: the model already rates its own rationales as highly likely (loss ~0.15), and only ~10% of examples correct an error. Preference training puts its signal on exactly the questions where samples disagree. **ScPO** (Prasad et al., ICML'25, arXiv:2411.04109) did this with self-consistency votes and roughly matched gold-label IRPO on GSM8K for Llama-3-8B. **What's new here:** the pseudo-label comes from the cross-format CoT+PAL vote (on run-1/2 data, its precision on questions with disagreement among the 5 CoT samples is 91.9% vs 84.4% for CoT-majority), on a 1.5B model, with a direct comparison against the CoT-only (ScPO-like) label and the gold upper bound.

## Design
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, Kaggle T4, vLLM for generation/eval (HF fallback).
- Train pool: GSM8K **train**, `random.Random(2029)` shuffle, first 3000 questions. This is a new shuffle, so the set partly overlaps run 2's questions; that's fine because it's train data.
- Per question: greedy CoT + 7 CoT samples (T=0.8), and greedy PAL + 4 PAL samples (T=0.8). Prompts and scorer are the same as in m3-selfdistill.
- Labels:
  - **MIX** (no gold): the plurality of all 13 answers. Kept if it has ≥7/13 votes and ≥1 CoT and ≥1 PAL vote.
  - **COT** (no gold, ScPO-like): the plurality of the 8 CoT answers. Kept if ≥5/8.
  - **GOLD** (upper bound, IRPO-like): the gold answer.
- Pairs: for each kept question, chosen = a distinct CoT text whose answer equals the label, and rejected = a distinct CoT text with a different or unparsed answer. Up to 2 pairs per question. All arms are subsampled to the same E pairs (≤2000).
- Training: LoRA r=16, α=32, all projections. Loss = DPO (β=0.1, reference = the base model with the adapter disabled) + 1.0 × length-normalised NLL on the chosen text (the RPO/IRPO recipe). 1 epoch, lr 5e-5, batch 16 pairs, warmup 10%, linear decay. Logged: DPO loss, NLL, pair accuracy, log-ratio margin.
- Eval: greedy CoT on the **full GSM8K test set (1319 q)**. Earlier fp16 work never trained or tuned on it; run 2's eval saw offsets 900–1318 but nothing was selected on them. Drift check: greedy CoT on BBH date_understanding (250 q, the bbh-mixvote2 prompt and parser), for every arm. Paired exact sign tests.
- In-job smoke pass of all phases (16 train q, 4 test q) before the full run.

## Prediction (written before any data from this run exists)
- **H1 (primary, confirmatory):** MIX − BASE on GSM8K test (n=1319). **Pass: ≥ +2.0 points AND sign-test p < 0.05.** Predicted **+2.5** (range −1 to +6). I give a full pass about 35%: run 2 showed this model is hard to move, but DPO targets the errors directly.
- **H2:** MIX − COT ≥ 0. Predicted +0.5. The two arms' labels differ on only ~3% of questions, so this will almost surely not be significant. Reported with p.
- **H3 (drift):** MIX − BASE on BBH date_understanding ≥ −3 points (no meaningful harm to a non-GSM8K skill). Predicted ~0.
- Reference (not pass/fail): GOLD − BASE predicted +3; COT − BASE predicted +2; BASE GSM8K greedy predicted 70% (range 66–74).
- Train pairs: MIX label precision on pair questions ≥ COT's. Predicted ~92 vs ~85.
- Anything else is exploratory.

## Runs
GPU job `run.py` via self/gpu_queue.txt (limit 330 min; expected ~3.5 h with vLLM). Outputs: `summary.json`, `train_stats.json`, `train_log.json`, `eval_<ARM>.jsonl` (GSM8K and BBH tails), `gen_answers.jsonl.gz`.

## Results
(pending)
