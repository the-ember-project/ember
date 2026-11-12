# math-g3-xscale: does a 1.5B model with G3 match a 3B model's greedy CoT on MATH?

Combining-pieces question from the vision: can cheap inference tricks on a small model replace a model twice its size?
Same 200 MATH numeric-subset test questions as `../math-g3-3b` (positions 2250–2449 of the `random.Random(2029)` shuffle),
now with Qwen2.5-1.5B-Instruct Q4_K_M: greedy CoT + 3 PAL samples (T=0.7), plurality vote, ties to CoT (`../math-g3/math_g3.py`).
Fresh sampling seed 9393. Compared, paired per question, with the 3B rows already in `../math-g3-3b/rows.jsonl` (3B greedy CoT 54.5%, 3B G3 63.0%).

Compute proxy: generated tokens × parameters (1.54B vs 3.09B). G3 at 1.5B used 2.34× CoT tokens on positions 2000–2249,
so 1.5B-G3 ≈ 1.2× the compute of 3B greedy CoT — roughly compute-matched.

## Prediction (written session #19, before any run)
- **Validity:** 1.5B greedy CoT boxed rate ≥ 85% and accuracy 35–65%; n=200 before any verdict.
- **H1 (non-inferiority, confirmatory):** acc(1.5B G3) ≥ acc(3B greedy CoT) − 3.0 points. Predicted 1.5B G3 ≈ 56%, 3B greedy 54.5% → **+1.5**.
- **H2 (superiority, confirmatory):** 1.5B G3 − 3B greedy ≥ +3 AND sign-test p < 0.05. Predicted to **fail** (+1.5, p large).
- **H3 (G3 replication at 1.5B, confirmatory):** 1.5B G3 − 1.5B greedy CoT ≥ +3 AND p < 0.05. Predicted +6 (math-g3 gave +6.4).
- Reported (exploratory): compute ratio (tokens×params) of 1.5B G3 vs 3B greedy, 3B G3 − 1.5B G3, per-level diffs.

## Results
(pending)
