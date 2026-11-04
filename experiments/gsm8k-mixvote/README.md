# Mixed-format voting (CoT samples + PAL programs) vs greedy CoT, GSM8K, Qwen2.5-1.5B-Instruct Q4_K_M

Why: on offsets 0–299, PAL alone is *worse* than CoT (61.7 vs 68.3, p=0.024; see `../gsm8k-pal/`), but when PAL and CoT agree (169/300) they are 94.1% correct, and on offsets 0–99 pass@5(CoT)+PAL = 89%. So a program is a partly independent "second opinion". This tests whether adding program answers to a CoT vote gives a significant gain over greedy CoT (roadmap M1). Related prior work: Zhao et al. 2023 "Automatic model selection with LLMs for reasoning" (CoT vs PAL selection), Self-consistency (Wang et al. 2022).

Dev evidence (offsets 0–99, exploratory, used to choose the selector — NOT part of the test): greedy 68, maj5 72, maj5+PAL(w=1) 74, **maj5+PAL(w=1.5) 76**, maj5+PAL(w=2) 72, maj4+PAL(w=1.5) 73. The weight was chosen among 3 values on 100 q, so expect some regression on fresh data.

## Design
Fresh questions: offsets **300–599** (n=300) of the seed-1234 shuffle, never used before. Fresh sampling seed: `--sample_seed 777` (earlier runs used 1234). All on the same questions (paired):
- greedy CoT (`../gsm8k-baseline/eval.py`, T=0)
- 5 CoT samples, T=0.7 (`eval.py --k 5`)
- greedy PAL (`../gsm8k-pal/pal.py`, T=0)
- 5 PAL samples, T=0.7 (`pal.py --k 5`)
Raw scorers only (no hand correction on any arm); exact numeric match |pred−gold|<1e-4.

## Prediction (written before any offset ≥300 data exists)
- **Primary (confirmatory), S1** = weighted vote over the 5 CoT samples (weight 1 each) + greedy PAL answer (weight 1.5); ties → earliest candidate (CoT samples in order, then PAL). Compared with greedy CoT on the same 300 questions.
  - Metric: accuracy difference S1 − greedy, discordant pairs (fixed/broken), exact two-sided sign test.
  - **Pass: S1 ≥ greedy + 5 points AND sign-test p < 0.05** (the roadmap M1 criterion). +3 to +5 or p ≥ 0.05 with diff ≥ +3 = "positive but not significant"; < +3 = fail.
  - Predicted: greedy 68% (63–73); S1 = greedy **+5** (range +2 to +8). I'd put ~35% on a full pass.
  - Cost: tokens per question of S1 / greedy; predicted ~5.5× (≈1,150 + 130 vs 233).
- **Secondary (confirmatory, separate criteria):**
  - S2 = plain majority over 5 CoT samples + 5 PAL samples (equal weight; ties → earliest, CoT first). Predicted greedy +5; pass same as S1. Predicted token cost ~7.5×.
  - S3 = maj@5 CoT alone (replication of the session-2 result on fresh data). Predicted greedy +3 (range 0 to +6); pass: ≥ +3 points (and p reported).
  - S1 − S3 (does adding PAL help beyond voting?): predicted +2; pass ≥ +2 points, sign test reported.
- Anything else (other weights, PAL-sample-only vote, agreement-based abstention) is exploratory.

Note (design, not a criterion change): S1 at ~5.5× tokens slightly exceeds the roadmap M1 cap of 5×; if S1 passes, M1 is met except for cost, and the 4-sample variant (exploratory) is the cost-compliant check.

## Runs
`run.sh` (lab, resumable). Greedy PAL for offset 300 may be partly run in-session (session #3).

## Results
(pending)
