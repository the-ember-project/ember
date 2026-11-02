# Self-consistency (majority vote) vs greedy on GSM8K, Qwen2.5-1.5B-Instruct Q4_K_M

First improvement experiment. Cheap idea: sample k=5 chain-of-thought answers at temperature 0.7 and take the majority answer (Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning"). Compare with the greedy baseline in `../gsm8k-baseline/` on the **same 100 questions** (sample seed 1234, offset 0), so the comparison is paired.

The lab job also runs greedy on 200 more questions (offset 100) to get a tighter baseline estimate (300 questions total).

Uses `../gsm8k-baseline/eval.py` with `--k 5 --temp 0.7`. Per-sample predictions are saved, so maj@1/maj@3/maj@5 can all be computed from the same run.

## Prediction (written before running)
- Hypothesis: majority voting over 5 samples beats greedy decoding for this small model.
- Metric: accuracy on the 100-question paired sample; also the count of questions greedy-wrong→vote-right vs greedy-right→vote-wrong; cost in tokens per question.
- Predicted: maj@5 is **+3 to +8 points** over greedy; costs ~5× tokens. A single temperature-0.7 sample (maj@1) is predicted to be *below* greedy by 0–5 points.
- Pass criterion: maj@5 accuracy ≥ greedy accuracy + 3 points on the same 100 questions. With n=100 this is a weak test (SE ≈ 4.7 points unpaired), so I will also report the paired discordant counts and treat a result inside ±3 as inconclusive rather than negative.

## Results
(Filled after the lab run.)
