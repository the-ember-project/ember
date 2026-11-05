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

## Results
(pending)
