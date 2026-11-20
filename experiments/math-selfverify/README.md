# math-selfverify: can Qwen2.5-1.5B grade its own MATH answers well enough to beat voting?

**Why.** On the math-sc7 items (fresh MATH test, n=250) 1.5B pass@7 is 78.0 but SC7 is 63.2. Exploratory analysis
(`experiments/math-select/explore.py`): of the 37 items SC7 misses while a correct sample exists, 30 have exactly **one**
correct sample out of 7 (8 of them with all answers distinct; 22 with a wrong plurality of 2–5). Length-based selectors
all lose to SC7 (1/tok weighting −1.6; shortest boxed −12.0). So recovering the gap needs a verifier that picks a lone
correct answer. math-verify (Kaggle, queued) tests a dedicated 1.7B ORM; this tests the free option: the policy itself,
on CPU, as a P(True)-style judge (Kadavath et al. 2022). Literature expectation: small models are yes-biased and weak
self-verifiers.

**Design.** For each of the 7 stored CoT samples (greedy + 6 at T=0.7, from math-mix7 `rows.jsonl` + math-sc7 `cs2.jsonl`),
prompt the same 1.5B Q4 model (llama.cpp) with the problem and the stored **300-char tail** of the solution (full texts were
not saved) and ask "Is the student's final answer correct? Yes or No". Score = P(Yes)/(P(Yes)+P(No)) on the first
token; use its log-odds. No prompt tuning (smoke on 3 items: P(Yes)≈0.98 everywhere — strong yes bias, as expected).
Selectors (all fixed now): **SVW** weighted vote, w_i = 2·sigmoid(logodds_i − median_j logodds_j); **SVtie** SC7 with ties
broken by highest log-odds; **BoN** highest log-odds sample. Limitation: tails only, so this is a lower bound on what
self-verification with full solutions could do. If math-verify's full texts come back, rerun on them with the same prompt
(minus "end part") for a direct comparison with the Skywork ORM.

## Prediction
- Metric: accuracy on the 250 items, paired vs SC7 (same samples), exact sign test; AUROC of log-odds for sample correctness.
- **H1 (signal):** pooled AUROC ≥ 0.60. Predicted **0.58, FAIL (~40% pass)**.
- **H2 (selection):** SVW − SC7 ≥ +2 points AND p < 0.05. Predicted **+0.0, FAIL (~10% pass)**.
- **H3 (tie-break):** SVtie − SC7 ≥ 0 (point estimate). Predicted **+0.4, PASS (~60%)**.
- Exploratory: BoN vs SC7 (predicted ≈ −8), within-question AUROC, mean P(Yes).
- Validity: Yes+No probability mass ≥ 0.9 on ≥ 95% of samples; all 250 items scored.
- Reading: H1 fail → self-P(True) from a 1.5B is no verifier; next is a dedicated ORM (math-verify) or a trained verifier.

## Results (lab run 2026-11-20, 114 min; confirmatory for H1–H3)
Validity PASS: all 250 items scored; min Yes+No mass 0.993. Mean P(Yes) 0.892 (strong yes bias, as in the smoke test).

| Hypothesis | Predicted | Measured | Verdict |
|---|---|---|---|
| H1 pooled AUROC ≥ 0.60 | 0.58 FAIL | **0.730** (853 pos / 897 neg) | **PASS** |
| H2 SVW − SC7 ≥ +2, p<.05 | +0.0 FAIL | **−0.8** (4 fixed / 6 broken, p=0.75) | FAIL |
| H3 SVtie − SC7 ≥ 0 | +0.4 PASS | **−0.4** (4/5, p=1) | FAIL |

Exploratory: BoN (top self-score) 55.6, −7.6 vs SC7 (5/24, p=5e-4); pass@7 78.0. Within-question AUROC 0.677 (139 mixed questions).

**Reading.** The 1.5B's P(Yes) carries real signal (H1 passed, against my prediction), but most of it is *question difficulty*, not which sample is right: post-hoc (`explore.py`, exploratory), the mean self-score predicts whether SC7 is right with AUROC 0.786, while within a question it separates right from wrong samples at only 0.677. On the 37 items that SC7 misses although a correct sample exists, the best correct sample is ranked first by self-score on 5/37 — about chance (30 of them have a lone correct sample, chance ≈ 1/7). So the self-judge cannot overrule wrong pluralities, and selection gains nothing. As a question-level confidence signal, vote agreement is better (AUROC 0.847); adding the self-score as a tie-break nudges it to 0.867 (exploratory, tuned-free but post-hoc).
Caveat: the judge saw only the stored 300-character tails. A full-text rerun on math-verify's outputs is the planned follow-up (direct comparison with the Skywork ORM on the same samples).
