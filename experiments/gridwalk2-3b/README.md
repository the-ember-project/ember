# gridwalk2-3b — does the program-format gain on gridwalk hold at 3B? (M2 scale check)

gridwalk2 (1.5B, n=200): greedy CoT 38.0%, single PAL sample 61.5%, G3 +33.5, vote(5P) ≈ mix → the program format
carries the gain. Question: does a bigger model (Qwen2.5-3B-Instruct Q4_K_M) close the CoT gap, making PAL/voting
matter less? Same task generator/prompts as gridwalk2 (10 steps, task seed 778, one-shot chat demo), **fresh indices
200–349 (n=150)**, PAL sample seed 93. Per question: greedy CoT, greedy PAL, 3 PAL samples T=0.7. CPU lab, resumable.

## Prediction (written before any run of this folder)
Verdicts only at n=150.
- Baseline: greedy CoT predicted 50% (range 30–70). Validity: greedy CoT skip rate (<30 tokens) ≤20%, else H1 not counted.
- **H1:** G3 = vote(greedy CoT + 3 PAL samples) − greedy CoT **≥ +3 AND p < 0.05** (exact sign test). Predicted +15 (range +3 to +30); ~85%.
- **H2 (format):** greedy PAL − greedy CoT **> 0 AND p < 0.05.** Predicted +10 (range −10 to +25); ~60%.
- Reported: vote(3P), vote(greedy PAL + 3P), PAL abstain rate, token cost.
