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

## Results (lab run 1, 2026-11-07, stopped at 290-min limit)
**0.5B: INVALID baseline — H1 not assessable.** Greedy CoT scored 4.3% (predicted 32–52%). Reading the outputs (session #10, prompted by the weekly review): with the shared system prompt ("Reason step by step, then give the final answer… '#### <number>'"), Qwen2.5-0.5B answers `#### <number>` immediately (e.g. 16 tokens) or loops on digits until the 400-token cap; it does no reasoning. So the "+21.8, 89/2" G3 number compares against a non-CoT baseline and is **not reported as a result**. Pilot on 10 different questions (offset 1200, not used for any test): original prompt 0/10 (~33 tok/q), tutor-style prompt (`--system`, see run2.sh) 7/10 at ~189 tok/q. Re-run with fixed prompt on fresh offsets: see Prediction v2.
**3B: interim, n=200 of 300 (not a verdict).** greedy 74.5%, G3 81.0%, +6.5, 17/4, p=0.007, cost 2.71×. vote(3 PAL) +3.5 (p=0.30). run2.sh resumes it to n=300; H2 is judged only at n=300.
Smoke-row note (review minor): the 2-question 3B smoke rows used identical deterministic settings (greedy; PAL seed = sample_seed·1000+j per question), so reuse by the resumable script cannot change outputs.

## Prediction v2 (0.5B, written session #10 before run2.sh runs)
Changes vs v1: 0.5B greedy uses system prompt "You are a helpful math tutor. Solve the problem step by step, writing out each calculation. Only at the very end, write the final answer on its own line as #### <number>."; fresh offsets 400–799 (n=400); fresh PAL seed 7171; files tagged `0.5b_v2`. PAL prompt unchanged (its v1 single-sample accuracy was 26.8%, unparsed 17.4% — seen, so PAL weakness is expected).
- **H1v2:** G3 − greedy **≥ +3 AND p < 0.05.** Predicted **+2** (range −2 to +6); ~35% chance. Greedy predicted ~38% (range 28–50). Reason for low prediction: G3 needs 2 agreeing PAL samples to overturn greedy, and 0.5B PAL is weak.
- Cost reported (≤3.5× expected).

