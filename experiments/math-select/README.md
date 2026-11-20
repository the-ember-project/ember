# math-select: training-free answer selectors over stored 1.5B MATH samples

Exploratory: post-hoc analysis of stored outputs (math-mix7 rows + math-sc7 cs2; fresh MATH test, n=250, 1.5B Q4).
No new generation. `python explore.py` → `explore.txt`.

| selector | acc | vs SC7 (fixed/broken) |
|---|---|---|
| SC7 (greedy + 6 sampled CoT) | 63.2 | — |
| pass@7 (oracle) | 78.0 | +14.8 |
| SC7, boxed answers only | 62.8 | −0.4 (2/3) |
| vote weighted 1/tokens | 61.6 | −1.6 (4/8) |
| vote weighted 1/sqrt(tokens) | 62.4 | −0.8 (4/6) |
| shortest boxed sample | 51.2 | −12.0 (9/39), p=2e-5 |
| MIX10 (SC7 + 3 PAL) | 65.2 | +2.0 (8/3) p=0.23 |
| SC7 + PAL at weight 0.5 | 65.6 | +2.4 (7/1) p=0.07 |

**Where SC7 loses pass@7 (37 items):** 30 have exactly 1 correct sample of 7, 3 have 2, 4 have 3. Of the 30 lone-correct
items, 8 have all answers distinct (a tie a verifier could break) and 22 face a wrong plurality of 2–5.
**Reading:** "shorter is better" does not hold for this 1.5B on MATH (the opposite: short answers are worse). Extra PAL
voters help a little (consistent with math-sc7: voter count matters). Closing the pass@k gap needs a verifier strong
enough to overrule a wrong plurality against a single correct sample; even perfect tie-breaking of the 12 tied items is worth at most +4.8.
Follow-ups: math-selfverify (CPU, self-P(True)), math-verify (Kaggle, 1.7B ORM).
