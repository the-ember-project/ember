# mix7-earlystop: lossless sequential stopping for the mix7 vote (analysis only, no new generation)

Session #49. Data: gsm8k-mixvote2 (1.5B Q4, offsets 600–899, n=300): 2 CoT + 5 PAL samples at T=0.7 per question.
Why: M5 caps inference at 1377 tok/q (5× 3B greedy). mix7 costs ~1200 tok/q for BASE and trained models write longer
CoT, so the cap may bind in m3-dpo-label Stage 4. A *lossless* rule (stop drawing voters once no completion of the
remaining votes can change the plurality winner, including the earliest-candidate tie-break) gives the identical answer
on every item, so accuracy is unchanged by construction and only tokens move.
Per-sample tokens are not stored separately: each sample is charged its question's mean for its format (CoT or PAL).

## Prediction
- Metric: mean tok/q of lossless-stop mix7 vs full mix7, on all 300 items; answers must be identical on 300/300 (check).
- Orders tried (vote list and tie-break order unchanged = cot1, cot2, pal1..pal5): (a) list order; (b) PAL first
  (pal1..pal5, cot1, cot2) — stopping still uses the list-order tie-break, so it is still lossless.
- E1: order (a) saves ≥25% tokens. Predicted ~30% (when all agree you stop after 4 of 7) → PASS.
- E2: order (b) saves more tokens than (a) (PAL samples are cheaper and agree often). Predicted PASS.
Label: confirmatory for E1/E2 (prediction written before computing); savings are specific to this model and slice.

## Results (confirmatory, session #49; `python analyze.py` → analysis.txt, results.json)
| order | tok/q | saving | mean voters drawn | answers identical | acc |
|---|---|---|---|---|---|
| full mix7 | 1113.0 | — | 7 | — | 73.0 |
| (a) list order | 903.9 | 18.8% | 5.29 | 300/300 | 73.0 |
| (b) PAL first | 803.8 | 27.8% | 5.31 | 300/300 | 73.0 |
- **E1 FAIL** (18.8% < 25%; predicted ~30%). **E2 PASS** (PAL-first saves 9 points more).
- Why E1 missed: on average 5.3 of 7 voters are needed, not ~4 — PAL non-executions (None) and the 1.5B's frequent
  2–3-way splits keep the vote open; only items where the first 4 voters agree stop at 4.
- Use: lossless PAL-first stopping cuts mix7 to ~0.72× tokens with the same answers, i.e. 1.5B mix7 ≈ 804 tok/q vs the
  M5 cap 1377 (2.9× 3B greedy). If a trained arm's mix7 exceeds the cap in m3-dpo-label, this rule can be applied to
  its stored per-item answers (token charge then approximate, since only per-format totals are stored). Known idea:
  Adaptive-Consistency (Aggarwal et al. 2023) stops sampling early with a lossy stability criterion; this is the
  exact (lossless) special case for a mixed-format vote.
