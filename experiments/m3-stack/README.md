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

## Results
(pending)
