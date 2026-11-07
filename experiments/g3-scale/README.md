# g3-scale — does the greedy-anchored cheap mix (G3) work at 0.5B and 3B?

G3 = vote(greedy CoT + 3 PAL samples at T=0.7, greedy wins ties). Confirmed for Qwen2.5-1.5B-Instruct Q4 on fresh GSM8K
(gsm8k-g3confirm: +6.9, p=0.0001, 2.65×). Roadmap item: second model size for the inference-time gains.

## Design
- Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct, Q4_K_M GGUF, llama-cpp (CPU lab). Same scripts and prompts as the
  1.5B runs (`gsm8k-baseline/eval.py`, `gsm8k-pal/pal.py`), GSM8K test shuffled with seed 1234.
- 0.5B: offsets 0–399 (n=400). 3B: offsets 0–299 (n=300), in chunks of 50 (resumable; may need a second lab run).
  These questions are fresh for these two models. PAL sample seed 6161 (new).
- `analyze.py <tag>`: paired sign test vs greedy, token cost.

## Prediction (written before running)
- **H1 (0.5B):** G3 − greedy **≥ +3 points AND sign-test p < 0.05.** Predicted +4 (range −2 to +9). Greedy ~42% (range 32–52).
  Risk: a 0.5B model may write poor programs (high abstention), in which case G3 falls back to greedy and gains little.
- **H2 (3B):** G3 − greedy **≥ +3 AND p < 0.05.** Predicted +4 (range 0 to +8). Greedy ~78% (range 70–86); less headroom.
- Cost ≤ 3.5× tokens for both (reported).
- Reference (not pass/fail): vote(3 PAL) alone and single PAL sample vs greedy; PAL unparsed rate.

## Results
(pending: lab)
