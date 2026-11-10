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

## Results (CPU lab 2026-11-10, 241 min, n=150 complete; CONFIRMATORY)
- **Validity PASS:** greedy CoT skip rate 0.0%; greedy CoT 41.3% (predicted 50).
- **H1 PASS:** G3 95.3%, **+54.0 (81/0), p<1e-4**, 3.9× tokens (predicted +15).
- **H2 PASS:** greedy PAL 96.7%, **+55.3 (83/0), p<1e-4**, 0.96× tokens (predicted +10).
- vote(3P) 95.3%, vote(greedy PAL+3P) 95.3%; PAL abstain 0%.
- Reading: at 3B the program format nearly solves 10-step gridwalk (1.5B: single PAL sample ~62%, vote 73%), while CoT barely improves with scale (38% → 41%). Scale helps *program writing* far more than *step-by-step simulation in text*. For this task one greedy program beats any vote — the cheapest method is just picking the right format.
