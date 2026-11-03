# Self-consistency (majority vote) vs greedy on GSM8K, Qwen2.5-1.5B-Instruct Q4_K_M

First improvement experiment. Cheap idea: sample k=5 chain-of-thought answers at temperature 0.7 and take the majority answer (Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning"). Compare with the greedy baseline in `../gsm8k-baseline/` on the **same 100 questions** (sample seed 1234, offset 0), so the comparison is paired.

The lab job also runs greedy on 200 more questions (offset 100) to get a tighter baseline estimate (300 questions total).

Uses `../gsm8k-baseline/eval.py` with `--k 5 --temp 0.7`. Per-sample predictions are saved, so maj@1/maj@3/maj@5 can all be computed from the same run.

## Prediction (written before running)
- Hypothesis: majority voting over 5 samples beats greedy decoding for this small model.
- Metric: accuracy on the 100-question paired sample; also the count of questions greedy-wrong→vote-right vs greedy-right→vote-wrong; cost in tokens per question.
- Predicted: maj@5 is **+3 to +8 points** over greedy; costs ~5× tokens. A single temperature-0.7 sample (maj@1) is predicted to be *below* greedy by 0–5 points.
- Pass criterion: maj@5 accuracy ≥ greedy accuracy + 3 points on the same 100 questions. With n=100 this is a weak test (SE ≈ 4.7 points unpaired), so I will also report the paired discordant counts and treat a result inside ±3 as inconclusive rather than negative.

## Results (lab run 2026-11-02, analysed session #2) — **confirmatory**
Analysis script: `analyze.py` (output in `analysis.txt`). The logged `acc` is maj@5 (majority over 5 extracted answers; checked: recomputed maj@5 matches the logged `ok` count). Both arms are scored with the same raw scorer (`####`, else last number).

| (same 100 questions) | accuracy | vs greedy: fixed / broken | exact sign test p |
|---|---|---|---|
| greedy CoT | 68% | — | — |
| single T=0.7 sample (mean of 5) | 63.2% | — | — |
| maj@1 (first sample) | 65% | 6 / 9 | 0.61 |
| maj@3 | 66% | — | — |
| **maj@5** | **72%** | **8 / 4** | **0.39** |
| pass@5 (any sample correct, oracle) | 87% | — | — |

Tokens: 1,165 per question vs 234 for greedy (5.0×). Lab speed 27.8 tok/s (lab machine is faster than the session sandbox).

- **Pass criterion (≥ +3): met (+4), but not distinguishable from noise** (8 fixed vs 4 broken, p = 0.39). Honest reading: consistent with a small gain; this test cannot show self-consistency helps.
- Single-sample prediction (0–5 below greedy): met (−4.8).
- Cost prediction (~5× tokens): met (5.0×).
- Scorer parity (review 2026-11-03): I checked all 28 maj@5-wrong questions for samples whose final sentence contained the gold number. One real scorer error (idx 1117, same as in greedy: "9 minutes to blink 459 times" scored 459); with it fixed, both arms gain one point (greedy 69, maj@5 73), so the margin and discordant counts are unchanged. I also checked all 57 greedy-correct answers lacking `####`: each ends in a concluding sentence stating the gold number, so no false credit found.
- **Exploratory observation:** pass@5 = 87% vs maj@5 = 72%. The right answer is often sampled but loses the vote, so a good *selector* (verifier, program check) has up to 15 points of headroom that voting does not use.
