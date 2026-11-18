# m3-stack: do label-free training and G3 stack to match the 2× bigger model?

**Why:** m3-vs-3b (#23) showed MIXR-DPO 1.5B closes 50% of the BASE→3B greedy gap on held-out GSM8K-train items
(81.7 → 87.1 vs 92.3), and xscale experiments showed G3 alone closes about half the gap at matched compute. On gridwalk,
self-training and G3 compounded (m3-gw-iter). Question: does MIXR + G3 (≈1.1× the compute of 3B greedy) reach 3B greedy?

**Design (Kaggle T4, `run.py`, ~130 min):** byte-for-byte copy of m3-dpo3/run.py (same gen seed 4041, same MIXR pair
rule, same DPO recipe, same 1500 held-out items idx[4500:6000], same eval/scorers), training only the MIXR arm
(pair count still equalised to the COT arm as in m3-dpo3), plus a new `g3` phase: for BASE and MIXR, 3 PAL samples per
item (T=0.7, same model/adapter as the greedy CoT), G3 = plurality over (greedy CoT, P1, P2, P3), ties to greedy CoT
(vote() identical to gsm8k-g3confirm). 3B greedy reference = stored m3-vs-3b/gpu-output/eval_BASE.jsonl (92.3%, same
items, same eval code). Comparison: `compare.py` (written after the job returns; paired sign tests on idx).
Compute ratio = (1.5 × tokens of greedy+3 PAL) / (3 × tokens of 3B greedy), from logged token counts.

## Prediction (written session #23, before any run)
- **Validity:** retrained MIXR greedy acc in 84.6–89.6 (m3-dpo3: 87.1; gen is not bit-reproducible); BASE greedy
  within ±1 of 81.7; PAL exec rate ≥ 90% for both arms. If MIXR falls outside, verdicts are marked compromised.
- **H1 (confirmatory, non-inferiority):** acc G3(MIXR) ≥ acc(3B greedy) − 2.0. Predicted G3(MIXR) ≈ 90.0 vs 92.3
  → **−2.3, FAIL** (~45% pass).
- **H2 (confirmatory, compounding):** G3(MIXR) − MIXR greedy ≥ +2.0 AND sign-test p < 0.05. Predicted **+3.0, pass** (~70%).
- **H3 (confirmatory, stacking):** G3(MIXR) − G3(BASE) ≥ +2.0 AND p < 0.05. Predicted G3(BASE) ≈ 85.5 → **+4.5, pass** (~80%).
- Exploratory: compute ratio (pred ~1.1×); G3 gain on BASE vs on MIXR; does CoT-only DPO change PAL accuracy?
  BBH-date drift of the retrained MIXR (m3-dpo3: −2.0).

## Prediction addendum (written session #24, 2026-11-15, before any run; job not yet started)
The weekly review (2026-11-15) pointed out that the held-out GSM8K-*train* items look contaminated/easy (BASE 81.7 there
vs 69.8 on test in m3-dpo; 3B 92.3 vs ~76), so "M3 met on GSM8K" may rest on a contaminated distribution. run.py now
also evaluates on the **full GSM8K test set (1319)**: greedy CoT + G3 for BASE and MIXR (`test_*` files), and 3B
greedy (same code, `--model Qwen/Qwen2.5-3B-Instruct`, no LoRA; `test3b_*`). H1–H3 above are unchanged.
- **Validity T:** BASE test greedy in 67–72 (m3-dpo BASE 69.8); 3B test greedy in 74–83.
- **H4 (confirmatory, the review's check):** MIXR − BASE greedy on GSM8K test ≥ +2.0 AND p < 0.05. Predicted **+2.5** (~50%).
- **H5 (confirmatory):** G3(MIXR) − G3(BASE) on test ≥ +2.0 AND p < 0.05. Predicted **+1.5, fail** (~40%).
- **H6 (confirmatory, non-inferiority):** G3(MIXR) ≥ 3B greedy − 2.0 on test. Predicted **−3, fail** (~35%).
- Exploratory: MIXR gain on test vs on held-out train (if test gain ≪ train gain, the train-split M3 result was inflated).
Smoke-tested the test-split eval and g3 phases on CPU (0.5B, HF, 2 items: item 0 = Janet's ducks, gold 18). Time
limit raised 160 → 195 min (extra ~30 min of vLLM eval).

## Results
(pending)

- Session #26 bug fix (no design change): eval phase now skips generation when the BBH list is empty (test-split and 3B steps use n_bbh=0); vLLM may reject an empty prompt list, which would have pushed those steps onto the slow HF fallback.

## Run 1 (Kaggle, 2026-11-17, ~206 GPU min): FAILED, no results
The in-job smoke pass caught a bug in the test-split eval: with no BBH items, `paired()` divided by zero (my session
#26 "skip empty BBH list" edit left `paired` on an empty list; it was never run on CPU after the edit). The crashed
phase process then **never exited** (vLLM engine shutdown hang), so the job sat until the 195-min limit. Nothing saved.
Fix (#29): `paired` returns None on empty input; every phase now hard-exits (`os._exit`) after catching any
exception; smoke steps have a 20-min timeout. Design and prediction unchanged. Re-queued `run.py 195`.
