# Confirming a cheap mixed vote (2 CoT + 5 PAL samples) vs greedy CoT — GSM8K, Qwen2.5-1.5B-Instruct Q4_K_M

Why: in `../gsm8k-mixvote/` (offsets 300–599) mixed CoT+PAL votes beat greedy CoT significantly, but the confirmed S1 cost 5.5× greedy tokens (over the M1 cap of 5×). A post-hoc cost curve (`../gsm8k-mixvote/costcurve.txt`) showed PAL samples are cheap (129 vs 226 tokens) and PAL-heavy mixtures win at equal cost: 2 CoT + 5 PAL = +10.0 at 4.8×; 1 CoT + 4 PAL = +8.0 at 3.2×. Those were picked among 35 configurations on the same data, so they need fresh questions.

## Design
Fresh offsets **600–899** (n=300) of the seed-1234 shuffle, never used. Fresh sampling seed **4242**. Paired arms: greedy CoT (`eval.py`), 2 CoT samples T=0.7 (`eval.py --k 2`), 5 PAL samples T=0.7 (`pal.py --k 5`). Raw scorers on every arm, exact match |pred−gold|<1e-4. Vote = equal-weight plurality over candidates in order [CoT samples..., PAL samples...], ties → earliest candidate (same `vote` as mixvote). Cost = mean generated tokens per question of the arms used ÷ greedy CoT tokens.

## Prediction (written before any offset ≥600 data exists)
- **Primary P1 (confirmatory):** vote(2 CoT + 5 PAL) vs greedy CoT. **Pass: diff ≥ +5 points AND exact two-sided sign-test p < 0.05 AND cost ≤ 5.0×** (= roadmap M1 in full). Predicted: greedy 67% (62–72); P1 **+7** (range +3 to +11), cost 4.8×. ~65% chance of a full pass.
- **Secondary P2:** vote(1 CoT + 4 PAL) vs greedy; same pass rule (cost ≤ 5×). Predicted **+5** (range +1 to +9), cost ~3.2×.
- **Secondary P3:** vote(5 PAL) alone vs greedy. Predicted **+2** (range −3 to +6); pass ≥ +3. Tests whether the mixture, not PAL sampling alone, carries the gain.
- **Secondary P4:** P1 vs vote(2 CoT) alone (should be ≈ greedy): predicted P1 better by ≥ +5, p < 0.05.
- Anything else is exploratory.

## Runs
`run.sh` (lab, resumable, two chunks of 150 so a partial run stays paired). Analysis: `python analyze.py`.

## Results (confirmatory; lab run 2026-11-05, offsets 600–899, seed 4242, n=300 paired, `analyze.py` output in lab-output.txt)
Greedy CoT 61.3% (predicted 62–72: just below range). Tokens/sample: greedy CoT 234, CoT sample 225, PAL sample 132.

| Arm | Acc | Diff | Fixed/broken | p (sign) | Cost | Verdict |
|---|---|---|---|---|---|---|
| **P1** vote(2 CoT+5 PAL) | 73.0 | **+11.7** | 49/14 | <0.001 | 4.75× | **PASS** (all three criteria; predicted +7) |
| P2 vote(1 CoT+4 PAL) | 72.3 | +11.0 | 46/13 | <0.001 | 3.22× | **PASS** (predicted +5) |
| P3 vote(5 PAL) | 66.7 | +5.3 | 40/24 | 0.060 | 2.82× | meets the ≥+3 threshold, but p=0.06: not distinguishable from noise at 0.05 |
| P4 P1 vs vote(2 CoT) | — | +9.7 | 39/10 | <0.001 | — | **PASS** (mixing, not sampling, carries the gain) |

**Roadmap M1 is now met in full:** ≥+5, p<0.05, ≤5× tokens on 300 fresh paired questions. Exploratory (post hoc mix sizes, same data): 1 CoT+2 PAL +9.7 at 2.1×; 1 CoT+3 PAL +11.7 at 2.7×; 2 CoT+3 PAL +12.3 at 3.6×. The curve flattens after ~3 PAL samples, and a single CoT sample plus 2–3 PAL samples gets most of the gain at 2–3× cost — a candidate default for later work (needs its own confirmation before being claimed).

## Exploratory addendum (session #7): greedy-anchored cheap mix (`routing.py`, `routing-output.txt`)
Checking the G3 rule found post hoc on BBH: vote(greedy CoT + 3 PAL), greedy wins ties: **+9.3, 33/5, p<0.001, 2.7×** (G2 +6.0 at 2.1×, G4 +10.0 at 3.3×). Slightly below vote(1 sampled CoT + 3 PAL) here (+11.7, 47/12), but with far fewer broken answers, and unlike that mix it does not collapse on BBH tasks where PAL fits badly. Exploratory.
