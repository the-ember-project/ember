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
**Complete, n=250 (session #28, one 243-min lab run; `analysis.txt`). Confirmatory.**
- **Validity PASS:** new sampled-CoT boxed rate 87.9% (≥80); SC4b 58.4 vs SC4 56.0 (+2.4, within ±4; 18/12, p=0.36).
- Accuracy: COT 51.6, G3 55.2, SC4 56.0, MIX7 60.4, **SC7 63.2**.
- **H1 FAIL (as predicted):** MIX7 − SC7 = −2.8 (7 fixed / 14 broken), p=0.19. Predicted +0.5; point estimate favours SC7.
- **H2 PASS:** SC7 − SC4 = +7.2 (20/2), p=1.2e-4. Predicted +3.
- Exploratory: SC7 − COT +11.6 (31/2, p=1.3e-7). Tokens vs greedy: MIX7 5.20×, SC7 7.00×.
- Exploratory, cost-matched (`costmatch.py`, written after seeing H1/H2): SC5 (greedy + 4 of the 6 sampled CoT, mean
  over all 15 subsets) 59.1 [57.6–61.2] at 5.0× vs MIX7 60.4 at 5.2×; SC6 61.5 at 6.0×.
- **Reading:** on MATH at 1.5B, more voters help a lot (SC4→SC7 +7.2). Format diversity adds nothing measurable: at
  equal voters single-format SC7 is ≥ MIX7 (−2.8 n.s.), and per token PAL voters are about as good as sampled CoT
  (MIX7 ≈ SC5). math-mix7's H3 (MIX7 > SC4) was voter count, not format mixing. Consistent with math-sc4 (G3 = SC4).


## Review 2026-11-22 follow-up (session #38, exploratory)
The review asked whether the 1024-token cap drove the voting gains. `trunc.py`: greedy truncated on 29/250 items;
SC7−COT is 3/0 on those (+1.2 pts) and 28/2 on finished items (+10.4 pts, p=9e-7); SC7−SC4 on finished items 17/2
(p=7e-4). Per-sample truncation ~10–11% (math-mix7 cs/cs2). So the gains here are not mainly truncation rescue, and my
session-35 remark that they were is withdrawn. At 2048 tokens (math-verify `voters.py`) SC8−SC4 is +4.1 (13/0) at 1.5B,
so "voter count matters" holds there too, though SC−greedy is much smaller on that slice (+2.8) for reasons not measured.
