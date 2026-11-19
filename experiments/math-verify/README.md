# math-verify: can a small verifier lift 1.5B best-of-8 to 3B level on MATH? (Kaggle, M5)

## Why
math-3b-sc (#31): on MATH, 1.5B SC7 (63.2) stays −4.8 below 3B greedy (68.0) and −5.2 below 3B SC3 at matched compute
(p=0.04). Yet 1.5B pass@7 on those items was **78.0**: the right answer is usually among the samples, and plurality
voting fails to pick it. Liu et al. 2502.06703 ("Can 1B LLM surpass 405B?") report small policies + PRM-guided search
beating far larger models on MATH-500, but with 7B–72B verifiers (often bigger than the policy) and without charging the
verifier's compute. Open question tested here: does a **small (1.7B) off-the-shelf outcome reward model**, counted in
the compute, close the 1.5B→3B gap? Verifier: Skywork-Reward-V2-Qwen3-1.7B (general ORM, `Qwen3ForSequenceClassification`).

## Design
- Items: MATH test (EleutherAI/hendrycks_math, numeric-gold subset, 4017 items), `random.Random(2029)` shuffle,
  positions **3700–4016 (n=317)**, never used before (2700–3699 are reserved for the m3-grpo main eval).
- Policies, vLLM fp16, SYS_COT (as m3-math-stack), max 2048 new tokens: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct,
  each greedy + 7 samples (T=0.7, top_p 1.0, seed 4242) = 8 per item.
- Scoring: Skywork-Reward-V2-Qwen3-1.7B (fp16; NaN scores re-scored in fp32), conversation = [user: problem,
  assistant: full response]. Exploratory second verifier: Skywork-Reward-V2-Qwen3-0.6B.
- Selectors over the 8 answers (extract = boxed number, else last number; answers None are abstentions):
  - SC8: plurality, ties to the earliest (greedy first).
  - **WSC8 (primary):** RM-weighted vote, weight = sigmoid(score); ties to the earliest.
  - BoN8: answer of the highest-scoring response (exploratory).
- Compute (FLOPs ≈ 2·params·tokens): 1.5B+RM8 ≈ 8×(1.54+1.72) = 26.1 per response-length unit; 3B SC8 ≈ 8×3.09 = 24.7
  (matched within 6%); 3B greedy = 3.09 (1.5B+RM8 is ~8.4× its compute).
- Stats: paired over items, exact two-sided sign test on discordant pairs.

## Prediction (written session #31, before any run)
- **Validity:** 1.5B greedy accuracy 50–65 and 3B greedy 61–76 (m3-math-stack, same setup: 55.5 / 68.6); unboxed rate
  of greedy ≤ 8% for both; RM separates right from wrong: AUROC of score vs correctness (1.5B samples) ≥ 0.60;
  NaN scores after fp32 retry 0. If a check fails, verdicts are reported but marked compromised.
- **H1 (verifier beats voting):** WSC8(1.5B) − SC8(1.5B) ≥ +2 points AND p < 0.05. Predicted **+2.0, borderline
  (~40% pass)** — general ORMs give modest gains over majority at N=8.
- **H2 (M5, non-inferiority vs 3B greedy):** WSC8(1.5B) − 3B greedy ≥ −2 (point estimate). Predicted **−3, FAIL
  (~40% pass)**; at 8.4× compute.
- **H3 (matched compute):** WSC8(1.5B) − SC8(3B) ≥ −2 (point estimate). Predicted **−6, FAIL (~15% pass)**.
- Exploratory: BoN8, pass@8, SC8 − greedy at both sizes, WSC8(3B), 0.6B RM, accuracy by MATH level.
- Reading: H1+H2 pass → a free 1.7B verifier substitutes for 2× scale on MATH (at high compute), a concrete M5 route.
  H1 fail → selection, not generation, is the bottleneck and off-the-shelf small ORMs don't fix it; a trained/self-
  generated verifier would be next.

## Run
`job.py 110` on Kaggle (one T4; estimated ~80 min from m3-math-stack throughput: 1.5B ~1900 tok/s, 3B ~700 tok/s). Phases run as subprocesses with timeouts and hard exits; a tiny in-job smoke
(2 items, 2 samples, 128 tokens, 0.6B RM) runs every phase first. Outputs in `gpu-output/`: `gen_{1.5B,3B}.jsonl.gz`
(full texts), `rm_{size}_{rm}.json` (scores). `python analyze.py` prints verdicts from the per-item files.
CPU smoke: `python job.py --smoke_cpu`.

## Results
(pending)
