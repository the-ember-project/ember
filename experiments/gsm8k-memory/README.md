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

## Results (session #20, n=300, two lab runs) — confirmatory NULL
Validity PASS: Z 64.7% (58–72); parsed 100% all arms; capped Z 4.3 / R 6.3 / X 8.0% (≤10); bank kept 251/400 traces
(label precision 94.4% vs 72.8% all-bank CoT accuracy).
- **H1 FAIL:** R 63.3 vs Z 64.7, R−Z **−1.3** (27 fixed / 31 broken), p=0.69. Predicted +1.
- **H2 FAIL:** R−X **+1.3** (32/28), p=0.70. Predicted +0.5.
- Exploratory: X−Z −2.7 (29/37, p=0.39). By top-retrieval-similarity tercile: low Z61/R61/X55, mid Z70/R64/X64,
  high Z63/R65/X67 — no sign that closer memories help more. Demos lengthen outputs (mean 231→252/264 tokens) and
  raise the cap rate; they mostly shuffle which questions are right (58 discordant R/Z pairs, net ≈0).
- Reading: for Qwen2.5-1.5B-Instruct on GSM8K, in-context recall of its own verified CoT traces does not help,
  even with a 94%-precise bank and TF-IDF retrieval. MoT's ChatGPT gains do not transfer to this setting. Caveats:
  one retriever (TF-IDF), 2 demos, a task the model was heavily tuned on. Weight-based memory (self-training, M3)
  worked where in-context memory did not.
