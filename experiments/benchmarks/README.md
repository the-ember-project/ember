# Benchmarks

| Name | Version | What it measures | Why | How |
|---|---|---|---|---|
| GSM8K-100 (public) | v1 | Grade-school multi-step math word problems; exact numeric answer | Well-known public benchmark, comparable with the field; small models make both reasoning and arithmetic errors, so it shows the effect of tools/voting | GSM8K `main` test, shuffle indices with `random.Random(1234)`, questions [0:100]. Extra fresh samples use later offsets ([100:300] etc.). Scorer: number after last `####`, else last number in output; exact match. Script: `experiments/gsm8k-baseline/eval.py` |

Known weaknesses: the fallback "last number" scorer can give credit to outputs without a clean final answer (rare); n=100 gives ±~5 pt standard error.

## Results log
| Date | Model | Method | Benchmark | Score | Notes |
|---|---|---|---|---|---|
| 2026-11-02 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy zero-shot CoT | GSM8K-100 v1 | 68% | 12.4 s/q on CPU; ~1% scorer error found (1 mis-score) |
