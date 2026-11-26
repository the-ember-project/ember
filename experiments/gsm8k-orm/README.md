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
`run_lab.sh` (resumable; ~3B greedy 1 h, 1.5B ×7 ~3.5 h, RM ~4 s per solution on the session CPU ≈ 1.9 h). Queued session #42 (`run_lab.sh 290`); finished in a second lab run (#43).

## Results (confirmatory, analysed session #44; `analysis.txt`)
Validity met: 3B greedy 74.8 (70–88), 100% of samples give an answer, 0 non-finite RM scores, n=210.

| arm | acc |
|---|---|
| 1.5B greedy | 68.1 |
| 1.5B SC8 | 73.3 |
| 1.5B WSC8 (1.7B RM) | 74.3 |
| 1.5B BoN8 | 77.1 |
| oracle@8 | 90.5 |
| 1.5B G3 (stored PAL) | 72.4 |
| 3B greedy | 74.8 |

- **H1 FAIL:** WSC8 − SC8 = +1.0 (3/1, p=0.63). Predicted +4. On GSM8K the RM barely changes the plurality.
- **H2 PASS (point estimate, as pre-registered):** WSC8 − 3B greedy = −0.5 (17/18, p=1). Predicted +1.
- **H3 FAIL:** same −0.5.
- Exploratory: SC8 − greedy +5.2 (p=0.019); 3B − 1.5B greedy +6.7 (p=0.049); SC8 − 3B −1.4 (n.s.); BoN8 − SC8 +3.8 (p=0.13).
  Compute: 1.5B×8 decode = 3.27× the 3B greedy decode (tokens × params), plus RM prefill.

**Caveat found at analysis (could change H2):** both arms ran with the pre-registered 400-token cap, but it bites
unequally: 3B greedy hit the cap on 23/210 items (19 of them wrong) vs 12/210 for 1.5B greedy (11 wrong). 3B's 74.8 is
far below its usual GSM8K score (85.2 on the full test in m3-stack, fp16, 2048 tokens), so H2's PASS may come from
truncating the more verbose 3B model. The H2 verdict stands as measured but is **not evidence for M5** until the
cap-fixed re-check below.

## Cap-fix re-check (prediction written session #44, before any re-generation)
Every output (3B greedy, 1.5B greedy, 1.5B samples) whose length is ≥395 tokens is *continued* from its truncated text
with the same model and temperature (greedy: T=0; samples: T=0.7, fresh seed) up to 1024 total tokens — equivalent to
having run with a 1024 cap. Changed 1.5B solutions are re-scored by the RM; others keep their scores. `uncap.py`,
`rescore.py`, `analyze.py --suffix _u`.
- Validity: continuation reproduces a held-back greedy tail on a smoke item; ≤2% of outputs still hit 1024.
- **H2u:** WSC8 − 3B greedy ≥ −2 (point estimate). Predicted **−3 → FAIL** (3B rises ~+5 to ~80, WSC8 ~+1).
- **H1u:** WSC8 − SC8 ≥ +2 and p<.05. Predicted +1 → FAIL.
- Reading: H2u FAIL → on GSM8K too, selection over 1.5B samples does not reach 3B greedy once 3B is not truncated.

### Cap-fix results (confirmatory, analysed session #45; `analysis_u.txt`)
Validity met: continued 22/210 3B greedy, 12/210 1.5B greedy and 56 sample rows; still at 1024: 0 / 1 / 1 row (≤2%).
- 3B greedy rose 74.8 → **79.5** (as predicted, ~+5); 1.5B greedy 69.0, SC8 73.8, WSC8 74.8, BoN8 77.1, oracle@8 91.0.
- **H2u FAIL:** WSC8 − 3B greedy = **−4.8** (11/21, p=0.11). Predicted −3, FAIL.
- **H1u FAIL:** WSC8 − SC8 = +1.0 (3/1). Predicted +1.
- Exploratory: 3B − 1.5B greedy +10.5 (p=0.0007); SC8 − 3B −5.7 (p=0.058); G3 − 3B −6.2 (p=0.06).
- Reading: the original H2 "PASS" was a truncation artefact. On GSM8K as on MATH, RM-weighted selection over 8 1.5B
  samples (3.25× decode compute) stays ~5 pts below 3B greedy. A free 1.7B RM adds ~+1 over plain voting on GSM8K.
