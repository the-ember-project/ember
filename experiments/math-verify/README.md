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

## Results (session #35; Kaggle run 2026-11-20, 77 GPU min) — CONFIRMATORY
n=317 fresh MATH test items (positions 3700–4016), 2048-token generations, greedy + 7 samples at T=0.7 per model.
**Validity: all passed.** Greedy 1.5B 54.9 (range 50–65), greedy 3B 67.2 (61–76); unboxed 7.6% / 1.9% (≤8%);
ORM AUROC on 1.5B samples 0.904 (≥0.60); non-finite scores 0.

| method | 1.5B | 3B |
|---|---|---|
| greedy | 54.9 | 67.2 |
| SC8 | 57.7 | 74.8 |
| WSC8 (Skywork-Reward-V2 1.7B, sigmoid-weighted vote) | **62.8** | 75.7 |
| BoN8 (1.7B ORM) | 63.7 | 74.1 |
| WSC8 / BoN8 with the 0.6B ORM | 61.8 / 63.7 | – |
| pass@8 | 75.4 | 87.4 |

- **H1 PASS:** WSC8 − SC8 (1.5B) = **+5.0** (17 fixed / 1 broken, p=1.5e-4). Predicted +2.0, borderline.
- **H2 FAIL:** WSC8(1.5B) − greedy 3B = **−4.4** (22/36, p=0.087); threshold −2. Predicted −3.
- **H3 FAIL:** WSC8(1.5B) − SC8(3B) = **−12.0** (6/44, p=3e-8). Predicted −6.
- Exploratory: BoN8 − SC8 +6.0 (p=0.004); the 0.6B ORM gives WSC8 +4.1 (p=0.004); at 3B the ORM adds only +0.9
  (n.s.); SC8 − greedy is only +2.8 at 1.5B (p=0.08) but +7.6 at 3B. Note that SC gains here (2048 tokens, vLLM) are
  much smaller than the +11.6 of math-sc7 (CPU, shorter budget): ~~part of the earlier SC gain was truncation rescue~~ — **wrong (review 11-22; corrected #38):** in math-sc7 only +1.2 of SC7's +11.6 sat on greedy-truncated items (`math-sc7/trunc.txt`). Exploratory `voters.py` (stored samples, 2048 tok): 1.5B SC1/SC4/SC8 54.9/53.6/57.7, SC8−SC4 +4.1 (13/0, p=2e-4), SC8−SC1 +2.8 (p=0.08); 3B 67.2/72.2/74.8, SC8−SC1 +7.6 (p=4e-5). Per-sample truncation 2.4% (1.5B) vs ~10–11% in math-mix7/sc7 at 1024. So voter count matters at 2048 too; why SC−greedy is smaller here than in math-sc7 is unexplained (different slice, fp16 vs Q4).
- `explore.py` (exploratory): **within-question AUROC 0.809** at 1.5B (0.782 at 3B), against 0.677 for the 1.5B
  self-judge (math-selfverify, tails only, different items). Of 56 SC8 misses with a correct sample, WSC8 rescues 17;
  of 33 with a lone correct sample, BoN picks it in 15. So a small external ORM *can* overrule wrong pluralities,
  which the self-judge could not.
- **N-curve (explore.py, exploratory; first k samples, greedy first):** 1.5B BoN-k 54.9/58.7/60.3/62.5/64.4/63.4/
  64.4/63.7 for k=1..8 — the 1.7B ORM's best-of-N plateaus at ~64 from k=5 while pass@k keeps rising (75.4 at 8);
  WSC-k rises slowly to 62.8. 3B: WSC-2 already 71.3. At roughly matched compute (3B k=2 ≈ 1.5B k=4–5) the 3B wins by
  ~8 points. Extrapolating, 1.5B + this ORM at N=16 would land ~64–65, below 3B greedy 67.2: **inference-time
  selection alone is unlikely to meet M5 on MATH; the ORM, not N, is the limit.** (Not tested; no N=16 job queued.)
- **Reading:** an off-the-shelf 1.7B (even 0.6B) reward model is the first selector that beats voting for the
  1.5B on MATH, by +5 points. It does not reach 3B greedy (−4.4 at ~8× generation + RM compute), and it does
  not help the 3B much. The remaining M5 gap is now ~4 points instead of ~12.

## Exploratory (session #36): ORM-guided adaptive stopping (`adaptive.py`, output `adaptive.txt`)
Sample sequentially (greedy first), stop once the leading answer's sigmoid(ORM)-weighted vote beats the runner-up by M.
On the same n=317: M=2 keeps the full WSC8 accuracy (62.8) at 6.0 vs 7.6 sample-units of generated tokens (−20%);
M=1 gives 62.1 at 4.9 units (−36%). 3B greedy gets 67.2 at 2.1 units (3B tokens counted 2×; ORM scoring cost not
counted, which favours the 1.5B). So on MATH, 3B greedy is cheaper than, and not worse than, every 1.5B+ORM operating point (review 11-22: the accuracy gap to WSC8 was −4.4, p=0.087, not significant): more accuracy at
less than half the compute. Swept thresholds on the evaluation items — exploratory only.
