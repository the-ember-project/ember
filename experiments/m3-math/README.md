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

## Results
(pending)
