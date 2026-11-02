# GSM8K baseline: Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp, CPU)

First number: how well does a small open model do on a well-known public benchmark on this machine, and how fast.

## Setup
- Model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- Runtime: llama-cpp-python 0.3.36 (CPU wheel), n_ctx 2048, 4 threads
- Data: GSM8K `main` test split (1319 questions). Random sample of N questions with `random.Random(1234)`.
- Prompting: zero-shot chat, system prompt asks to reason step by step and end with `#### <number>`. Greedy decoding (temperature 0), max 400 new tokens.
- Scoring: extract number after the last `####`; fallback = last number in the output. Exact numeric match with the gold answer.

## Prediction (written before running)
- Metric: accuracy (exact match) on the 100-question sample; speed in generated tokens/s and seconds/question.
- Qwen reports ~73% GSM8K for Qwen2.5-1.5B-Instruct (full precision, their prompting). With Q4 quantisation and my simple zero-shot prompt I predict **55–70%** accuracy.
- Speed prediction: 10–25 tokens/s generation, ~10–20 s per question.
- Pass criterion for "baseline is usable": accuracy ≥ 50% and ≤ 30 s/question (so 100 questions fit in under an hour of lab time). This is a baseline, not a hypothesis test.

## Results (session #1, 2026-11-02) — confirmatory against the prediction above
- **Accuracy: 68/100 = 68%** (inside predicted 55–70%). 95% CI roughly 58–77%.
- **Speed: 18.8 generated tok/s, 12.4 s/question** (inside predicted 10–25 tok/s, 10–20 s/q). 100 questions ≈ 21 min on 4 threads.
- Pass criterion (≥50%, ≤30 s/q): **met** on both.
- Comparison: Qwen's reported 73.2% (full precision, their prompt) — Q4 + my zero-shot prompt is ~5 points lower, within noise of n=100.
- Raw outputs: `results_qwen1.5b_greedy_n100.jsonl`.

Observations (exploratory):
- Format compliance is poor: only 11/100 outputs used `#### <n>`; 15 used `\boxed{}`; most end "Therefore ... is N." Scoring relied mostly on the last-number fallback.
- Manual check of the 32 wrong answers for scorer errors: 1 clear mis-score (idx 1117: answer "9 minutes to blink 459 times" → scored 459), 1 sign case (idx 1044: "-1 inch" vs gold 1, kept as wrong). So the true accuracy is ~69%; scorer error ≈1%. Logged as a benchmark weakness; a v2 scorer (prefer `\boxed`/`####`, then the number right after "is" in the last sentence) can come later if it matters.
- 2/100 hit the 400-token limit.
