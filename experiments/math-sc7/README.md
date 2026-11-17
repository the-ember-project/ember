# math-sc7: does format diversity help at equal voter count on MATH?

math-mix7 (#27, confirmatory, n=250): MIX7 (greedy + 3 sampled CoT + 3 PAL) beat SC4 (greedy + 3 sampled CoT) by +4.4
(p=0.013) and G3 by +5.2. But MIX7 has 7 voters and SC4 has 4, so "more voters" and "mixed formats" are confounded.
At 4 voters, format didn't matter (G3 ≈ SC4 in math-sc4 and math-mix7). This adds 3 more sampled CoT (new seeds) on the
same 250 items so SC7 (greedy + 6 sampled CoT) can be compared with MIX7 at 7 voters each.

Prior work: Self-consistency (Wang 2022) gains grow with samples; XoT / Zhao 2305.14333 mix CoT and PAL. My GSM8K errcorr
found CoT/PAL errors less correlated than CoT/CoT. On MATH the PAL exec rate is only 62%, so PAL voters often abstain.

## Prediction (written session #27, before any run)
- **Validity:** new sampled-CoT boxed rate ≥ 80%; n=250; SC4b (greedy + the 3 new samples) within ±4 of SC4 (seed sanity).
- **H1 (confirmatory):** MIX7 − SC7 ≥ +2 points AND sign-test p < 0.05. Predicted **+0.5, FAIL (~20% pass)**:
  PAL voters abstain 38% of the time, so SC7 has more effective voters; on MATH PAL was not a better voter than sampled CoT.
- **H2 (confirmatory):** SC7 − SC4 ≥ +3 AND p < 0.05. Predicted **+3, ~45% pass** (MIX7 − SC4 was +4.4).
- Exploratory: SC7 − COT; tokens (predicted SC7 ≈ 7×, MIX7 ≈ 5.2×). If H1 fails with SC7 ≥ MIX7, the MATH reading is:
  "more voters help; PAL voters are cheaper, not better". If H1 passes, format diversity matters beyond voter count.

## Run
CPU lab `run_lab.sh 290` (resumable, ~55 s/q expected → ~230 min). `python analyze.py` for verdicts.

## Results
(pending)
