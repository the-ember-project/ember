# math-3b-sc: spend compute on votes (1.5B SC7) or on scale (3B)? (CPU lab, M5 on MATH)

On math-mix7's 250 fresh MATH items (2029 shuffle, positions 2450–2699), 1.5B SC7 (greedy + 6 sampled CoT) scored 63.2
and MIX7 60.4 (math-sc7). Here Qwen2.5-3B-Instruct Q4_K_M answers the same items with the same prompt (m3-math
SYS_COT), 1024-token limit, n_ctx 2048: phase 1 greedy CoT, phase 2 two sampled CoT (T=0.7, seed 6262). 3B SC3
(greedy + 2 sampled) costs about the same as 1.5B SC7 in tokens × params (≈3 × 3.09B vs 7 × 1.54B), so the two
tests are: (H1) does 1.5B with 7 voters reach 3B greedy at all (≈3.5× its compute), and (H2/H3) at matched compute,
is the bigger model better? Prior: math-g3-xscale (other items) had 3B greedy 54.5 vs 1.5B greedy 44.5 (+10.0) and
1.5B G3 −5.5 vs 3B greedy (n.s.). Snell 2408.03314: test-time compute beats a larger model only where the small model
already partly solves the problem; Wu 2408.00724: small models + inference methods can be compute-optimal.

## Prediction (written session #28, before any run)
- **Validity:** 3B greedy boxed rate ≥ 85% and 3B greedy accuracy 52–70; n=250 greedy and n=250 sampled before the
  H2/H3 verdicts (H1 may be reported once greedy reaches 250).
- **H1 (non-inferiority, confirmatory):** 1.5B SC7 − 3B greedy ≥ −2 points (point estimate). Predicted 3B greedy ≈ 61,
  **+2, PASS (~65%)**.
- **H2 (scale wins at matched compute, confirmatory):** 3B SC3 − 1.5B SC7 ≥ +3 AND sign-test p < 0.05. Predicted
  3B SC3 ≈ 64, **+1, FAIL (~25% pass)**.
- **H3 (non-inferiority at matched compute, confirmatory):** 1.5B SC7 − 3B SC3 ≥ −2. Predicted −1, **PASS (~55%)**.
- Exploratory: compute ratios (tokens × params), 3B SC3 − 3B greedy, MIX7/G3(1.5B) − 3B greedy, 3B − 1.5B greedy.
- Reading: H1 pass + H3 pass → on MATH, voting at 1.5B can substitute for 2× scale at matched compute (M5-relevant,
  though it's MATH with simple voting). H2 pass → scale still wins at matched compute.

## Run
CPU lab `run_lab.sh 290`, resumable; ~47 s per 3B generation on the lab → phase 1 ~200 min, phase 2 ~400 min
(≈3 lab runs). `python analyze.py` prints verdicts (partial results print without H2/H3).

## Results (session #31, complete: greedy 250/250, sampled 250/250; confirmatory)
- **Validity PASS:** 3B greedy boxed 91.6% (≥85), accuracy 68.0 (52–70); 3B sampled boxed 90.6%.
- Accuracy: 1.5B COT 51.6, G3 55.2, MIX7 60.4, SC7 63.2; 3B COT 68.0, **3B SC3 68.4**.
- **H1 FAIL:** 1.5B SC7 − 3B greedy = −4.8 (14/26), p=0.08, at 3.38× compute. Predicted +2.
- **H2 PASS:** 3B SC3 − 1.5B SC7 = **+5.2** (24 fixed / 11 broken), p=0.041, at matched compute (SC7/SC3 = 1.10×).
  Predicted +1 (FAIL).
- **H3 FAIL:** 1.5B SC7 − 3B SC3 = −5.2, below the −2 margin. Predicted −1 PASS.
- Exploratory: 3B SC3 − 3B greedy = +0.4 (4/3): two extra samples at 3B barely change the vote (with 3 voters the
  greedy answer wins most ties), so at 3B the matched-compute advantage is essentially *greedy 3B* beating 7-voter 1.5B.
  3B − 1.5B greedy = +16.4 (50/9).
- **Reading:** on MATH at this pair of sizes, scale beats simple voting at matched compute (one test, p=0.04, n=250).
  The 1.5B→3B gap (16 pts) is ~3× what 7-voter SC recovers (11.6). Consistent with Snell 2408.03314 (test-time compute
  only substitutes for scale on problems the small model partly solves). M5 via voting alone looks out of reach on MATH.

## Results so far (session #29, after lab run 1: greedy 250/250, sampled 27/250)
- **Validity (greedy):** 3B greedy boxed 91.6% (≥85 PASS), accuracy 68.0 (52–70 PASS).
- **H1 FAIL (confirmatory):** 1.5B SC7 63.2 − 3B greedy 68.0 = **−4.8** (14 fixed / 26 broken), p=0.08, at 3.4× the
  compute (tokens × params). Predicted +2. Even 7 voters at 1.5B do not reach 3B greedy on MATH.
- Exploratory: MIX7 − 3B = −7.6 (p=0.005); G3 − 3B = −12.8; 3B − 1.5B greedy = +16.4 (50/9).
- H2/H3 wait for sampled 250/250 (re-queued `run_lab.sh 290`).
