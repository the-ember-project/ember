# m3-math: label-free cross-format DPO on a public benchmark beyond GSM8K (MATH, numeric answers)

**Why.** M3 is met on GSM8K (MIXR DPO +4.2, +5.3) and on my own gridwalk task (PALV SFT +16.8; iterates: m3-gw-iter
R2 +5.2 over R1; compounds with G3 +11.0). The open question is whether *cross-format* label-free self-training
works on a harder, public, non-synthetic benchmark where CoT is weaker and programs (sympy/fractions) can help.
MATH (Hendrycks) has a train split, so training and evaluation are disjoint.

**Prior work (arXiv API check, session #16).** T³RL (arXiv 2603.02203) grounds test-time-RL majority-vote
pseudo-labels in code-execution evidence (verification-aware voting) on MATH-500/AMC/AIME, i.e. tool evidence improves
label-free pseudo-labels. SCRL (2603.19880) and DDRL (2604.21327) filter weak consensus. Mixed Distillation
(2312.10730) distils CoT+PoT from a *teacher LLM*. My version differs: offline DPO (not RL), the teacher is the
model's *own* program samples pooled into the vote, 1.5B, CPU-runnable result. Not new in spirit; a re-test on a
small model with a cheap recipe.

## Design (copy of m3-dpo3's recipe; only the data, prompts and parser change)
- Model Qwen2.5-1.5B-Instruct fp16, Kaggle T4, vLLM generation, HF+PEFT LoRA DPO+NLL (β=0.1, α=1, lr 5e-5, 1 epoch,
  r=16), up to 2 pairs per question, equal pair count E = min(cap 1200, |MIXR|, |COT|).
- Data: MATH all 7 subjects, only problems whose `\boxed{}` gold parses as a number (int/decimal/fraction): train
  5.9k, test 4.0k. Generation on 2000 train problems (shuffle seed 2029): greedy + 7 CoT + greedy + 4 PAL at T=0.8,
  vLLM seed 6061, max 1024 new tokens (raised from 768 after a 2-question CPU smoke truncated both CoTs; no results seen), DPO max_len 1200. CoT prompt asks for `\boxed{answer}`; PAL may use math/fractions/sympy.
- Labels (label-free): **MIXR** = plurality ≥5 of 13 answers (8 CoT + 5 PAL) and present in both formats;
  **COT** = CoT majority ≥5 of 8. Chosen = CoT trace with the label answer, rejected = CoT trace without it.
- Eval: greedy CoT on 1200 MATH-test numeric problems (shuffle seed 2029) per arm; BBH date (n=250) drift.
  Scoring: parse `\boxed{}` number (fallback: last number), match with relative tolerance 1e-4.
- `python run.py` (Kaggle) smoke-tests every phase on 16 problems before the full run. CPU smoke (0.5B, HF, doctored
  labels for train) passed all three phases in session #16.

## Prediction (written session #16, before any run)
- **Validity check:** BASE greedy accuracy in 35–65%, and BASE outputs with no `\boxed{}` (truncation/format failure)
  ≤ 10%. If either fails, the H verdicts are reported but marked compromised.
- **H1 (confirmatory):** MIXR − BASE on MATH test ≥ +2.0 AND sign-test p < 0.05 (n=1200). Predicted **+2.0**
  (range −3 to +6); ~45% (harder task, noisier labels than GSM8K; Qwen2.5 was heavily trained on MATH).
- **H2:** COT − BASE ≥ +2.0, p<0.05. Predicted ~0 (range −6 to +3); expected FAIL (~25%).
- **H3:** MIXR − COT ≥ 0 (point estimate). Predicted pass (~65%).
- **H4 (drift):** MIXR − BASE on BBH date ≥ −3 (point estimate). Predicted pass (~55%).
- Reported, no criterion: label precision of MIXR vs COT on pair questions; per-subject and per-level diffs.
- Verdicts computed from `eval_*.jsonl` (`correct`, `boxed` fields), not only summary.json.

## Runs
Kaggle via `self/gpu_queue.txt`, limit 290 min (estimate ~190: gen ~70, train ~70, eval ~25, setup+smoke ~20).
- **Run 1 (2026-11-13, 292 GPU min): crashed, no results.** vLLM gen ran 7150 s (not ~70 min), then `num()` raised OverflowError on a huge fraction in a PAL output. The HF fallback then OOMed until the limit. Nothing was evaluated. Fix (session #21): `num()` and `run_code()` now return None on any parse/overflow error or non-finite value (unit-tested on a 400-digit fraction and on 10**400). Design and prediction unchanged. Re-queued at 340 min.

## Results (run 2, 2026-11-14/15, 351 GPU min, vLLM throughout; computed from eval_*.jsonl, matches summary.json)
Pairs: MIXR 1090 pairs on 661 questions (label precision 90.2%), COT 1069 pairs on 670 q (precision 91.8%); E=1069.
Greedy CoT, MATH test numeric, n=1200 paired:

| arm | acc | boxed | BBH date (n=250) |
|---|---|---|---|
| BASE | 54.83 | 86.6% | 49.6 |
| MIXR | 58.58 | 91.5% | 48.4 |
| COT  | 59.17 | 91.8% | 44.0 |

- **Validity: accuracy PASS (54.8 in 35–65); unboxed-rate FAIL (13.4% > 10%)** → verdicts below are **compromised**.
- **H1 PASS (compromised):** MIXR − BASE **+3.75** (119 fixed / 74 broken), p=0.0015. Predicted +2.0.
- **H2 PASS (compromised), predicted FAIL:** COT − BASE **+4.33** (125/73), p=0.0003.
- **H3 FAIL:** MIXR − COT −0.58 (82/89), p=0.65.
- **H4 PASS:** BBH-date drift MIXR −1.2 (25/28, p=0.78); COT −5.6 (p=0.11, no criterion).
- Per level (BASE/MIXR/COT): L1 80.4/85.7/85.7, L2 74.5/75.9/78.6, L3 62.5/67.5/69.6, L4 52.0/54.6/55.0, L5 24.7/29.7/27.6.

**Exploratory decomposition (why validity matters here).** On the 161 items where BASE gave no `\boxed{}` (mostly
runs past 1024 tokens), MIXR fixes 28 / breaks 2 (p<1e-6) and COT 31/6: DPO teaches the model to *finish and box*.
That accounts for 2.2 of MIXR's 3.75 points (58%) and 2.1 of COT's 4.33. On the 1039 items BASE boxed, MIXR is
+1.6 (91/72, p=0.16) and COT +2.25 (94/67, p=0.040). So about half the gain is format/termination, half reasoning,
and only the COT arm's reasoning part is significant.

**Reading.** Label-free DPO on own outputs helps 1.5B on public MATH (+3.75 to +4.3, both p<0.002), but here the
*cross-format* label adds nothing: CoT-majority-of-8 labels are as precise as MIXR labels (91.8 vs 90.2%), unlike
GSM8K (m3-selfdistill: MIX 96.7 vs COT 92.6) where same-format DPO was unstable. Fits math-sc4 (PAL voters buy cost,
not accuracy, on MATH). The rule "a different format as teacher is what makes label-free self-training work" is
therefore **task-dependent**: it holds where the program format is a stronger, more independent solver (GSM8K
arithmetic, gridwalk), not on MATH. One training run per arm; same-format DPO gave −7.9 on one GSM8K draw, so H2 is
one draw, not a stable result.
