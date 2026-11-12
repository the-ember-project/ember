# gsm8k-memory: Memory-of-Thought at 1.5B with a cross-format filter

**Question.** Does a small model reason better when it can recall its own earlier *verified* solutions to
similar problems? This is the "memory of what worked" piece of the roadmap vision, untested so far.

**Prior work.** MoT (Li & Qiu 2023, arXiv 2305.05181): ChatGPT pre-thinks on unlabeled data, stores
high-confidence (self-consistency) thoughts, recalls relevant ones as demos at test time; reports significant
gains. eMoT (2606.02054) adds Python anchoring and memory decay. New here: a 1.5B Q4 model on CPU, a
*cross-format* filter (CoT answer = program answer, which gave 87–97% label precision in my M3 work), and a
random-demo control that separates "retrieval of similar problems" from "any self-demos".

**Design.** Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp, greedy, max 400 tokens.
- Bank: 400 GSM8K *train* questions (seed 77). Greedy CoT + greedy PAL; keep CoT traces whose answer equals
  the PAL answer (no labels used; gold only for reporting precision).
- Eval: GSM8K test, seed-1234 shuffle, offsets 300–599 (n=300). Per question three greedy-CoT arms with the
  same system prompt: **Z** zero-shot; **R** 2 TF-IDF-nearest bank traces as prior chat turns (most similar
  last); **X** 2 random bank traces (seed from question idx).
- Paired sign test (exact, two-sided) on discordant pairs. `mem.py` (resumable), `analyze.py`.

## Prediction (written session #18, before any run)
- **Validity:** Z accuracy in 58–72% (earlier greedy on 300–599: 65.3%); in every arm a number is parsed from ≥95% of
  outputs and ≤10% of outputs hit the 400-token cap. (Edited before the real run: the smoke test showed the
  model often ends without `####`, so a `####` filter/threshold was dropped.) Bank keeps ≥150 traces. Otherwise the verdict is marked compromised.
- **H1 (confirmatory):** R − Z ≥ +3.0 points AND sign-test p < 0.05 (n=300). Predicted **+1** (range −3 to +4);
  I expect H1 to FAIL: Qwen2.5-Instruct is heavily tuned on GSM8K-style CoT, so its own demos add little.
- **H2 (confirmatory):** R − X ≥ +2.0 points (retrieval beats random self-demos). Predicted **+0.5**; expect FAIL.
- Reported (exploratory): bank size and precision, X − Z, accuracy by retrieval-similarity tercile, prompt tokens.
- No verdict before n=300.

## Results
(pending)
