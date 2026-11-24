# gsm8k-orm: does a free 1.7B reward model let 1.5B match 3B greedy on GSM8K?

**Why.** On MATH (math-verify, #35) Skywork-Reward-V2-Qwen3-1.7B weighting of 8 1.5B samples gave +5.0 over SC8 but
stayed −4.4 below 3B greedy; best-of-N saturated below 3B. MATH's 1.5B→3B gap is 13–16 pts. On GSM8K the gap is
smaller (68.3 vs 75.7 on offsets 0–299, gsm8k-xscale) and voting gains are larger, so selection alone might reach 3B
greedy there. This is the second-benchmark data point for M5's "accuracy match" (not compute-matched: 8 × 1.5B decode
+ 1.7B prefill vs one 3B decode; ratio reported) and a CPU-only counterpart to the GPU job m3-grpo-orm's O4.
Uses the free CPU lab while GPU quota is the bottleneck.

## Design
- GSM8K test, `random.Random(1234)` shuffle, offsets 900–1109 (n=210). 1.5B greedy CoT is stored
  (gsm8k-g3confirm/cot_greedy_off900.jsonl, first 210 lines); the RM, the 7 samples and the 3B outputs are new.
- 1.5B: greedy + 7 CoT samples (eval.py prompt, T=0.7, **fresh sample_seed 6262**, max 400 tokens), Q4_K_M llama-cpp.
- 3B greedy: Qwen2.5-3B-Instruct Q4_K_M, same prompt/scorer.
- RM: Skywork-Reward-V2-Qwen3-1.7B, CPU fp32, batch 1, chat template (question, solution) → scalar.
- SC8 = plurality of 8 answers (ties → earliest, i.e. greedy). WSC8 = plurality weighted by sigmoid(RM score)
  (math-verify protocol). Paired exact sign tests.

## Prediction (written session #42, before any run)
- Validity: 3B greedy 70–88%; ≥95% of samples yield an answer; 0 non-finite RM scores; n=210 paired before verdicts.
- **H1 (RM helps selection on GSM8K):** WSC8 − SC8 ≥ +2 AND p<.05. Predicted **+4, PASS** (SC8 ≈ 73, WSC8 ≈ 77).
- **H2 (M5-style accuracy match, non-inferiority, point estimate):** WSC8(1.5B) − 3B greedy ≥ −2. Predicted **+1, PASS**
  (3B greedy ≈ 76).
- **H3 (superiority):** WSC8(1.5B) − 3B greedy ≥ +3 AND p<.05. Predicted **FAIL**.
- Exploratory: SC8 − greedy, BoN8, oracle@8, G3 (stored PAL) vs 3B, compute ratio.
- Reading: H2 PASS on GSM8K but FAIL on MATH → selection substitutes for 2× scale only where the gap is small/the
  answer is checkable; H2 FAIL on both → selection alone does not reach the next size up at 1.5B.

## Run
`run_lab.sh` (resumable; ~3B greedy 1 h, 1.5B ×7 ~3.5 h, RM ~4 s per solution on the session CPU ≈ 1.9 h). Queued session #42 (`run_lab.sh 290`); needs a second run — re-queue.
