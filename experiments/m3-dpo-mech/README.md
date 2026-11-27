# m3-dpo-mech: where do the label-free MIXR-DPO gains on GSM8K come from?

Analysis only, no new generation (session #47). Data: m3-stack (seed R1) and m3-dpo-rep (R2, R3) greedy
evals on the full GSM8K test (n=1319), plus m3-stack `test_g3_BASE.jsonl.gz`: BASE greedy CoT + 3 sampled
PAL answers per test item (the test-time analogue of the MIXR training label source).

Question: MIXR-DPO trains toward the model's own CoT+PAL vote. Are its test gains (a) concentrated on items where
the BASE model's own vote/samples already contain the right answer ("sharpening", Lin et al. 2605.19444), and
(b) does it import the vote's *errors* (does it break items in the direction of the wrong vote)?

Strata over BASE-greedy-wrong items (by BASE samples on the test item):
- A: G3 (greedy + 3 PAL vote) correct.
- B: G3 wrong but ≥1 PAL sample correct.
- C: no sample (greedy or PAL) correct.
Over BASE-greedy-correct items: AG = G3 correct, WG = G3 wrong (vote disagrees with correct greedy).

Unit: item × seed (3 seeds). Fix rate = P(MIXR correct | BASE wrong); break rate = P(MIXR wrong | BASE correct).

## Prediction (written before computing any stratified number)
- P1: fix rate in A ≥ 2× fix rate in C. Predict A ~40%, C ~12% → PASS.
- P2: break rate in WG ≥ 2× break rate in AG. Predict WG ~25%, AG ~6% → PASS.
- P3: share of all fixes (item×seed) coming from C ≤ 30%. Predict ~25% → PASS.
- P4: among breaks in WG, the share where MIXR's new answer equals BASE's G3 (wrong) answer ≥ 50%. Predict ~45% → FAIL
  (narrow; the vote's wrong answer is one of many possible wrong answers).
Caveat fixed in advance: P1 and P3 cannot separate "learned from the vote" from "any perturbation fixes easy items"
(A items are easier). P4 is the more specific test of imitation of the label source. 3 PAL samples is a weak
probe of what the model "can" solve, so C overstates truly-unsolvable items.
Label: confirmatory for P1–P4 as written; anything else exploratory.

## Results (session #47; confirmatory for P1–P4, consistency.py exploratory)
`python experiments/m3-dpo-mech/analyze.py` → `analysis.txt`; `python consistency.py` → `consistency.txt`.
Strata from m3-stack's BASE samples; R2/R3 use their own BASE greedy (17/22 of 1319 greedy answers differ from
m3-stack's BASE run: GPU nondeterminism; correctness differs on 7).

| stratum (BASE) | item×seed | MIXR changed | rate | net items/seed |
|---|---|---|---|---|
| A: greedy wrong, G3 right | 332 | 181 fixed | 54.5% | +60.3 |
| B: G3 wrong, a PAL sample right | 290 | 113 fixed | 39.0% | +37.7 |
| C: no sample right | 577 | 114 fixed | 19.8% | +38.0 |
| AG: greedy right, G3 right | 2668 | 232 broken | 8.7% | −77.3 |
| WG: greedy right, G3 wrong | 90 | 24 broken | 26.7% | −8.0 |

- P1 PASS: fix rate A/C = 2.76 (pred ~3.3).
- P2 PASS: break rate WG/AG = 3.07 (pred ~4).
- P3 PASS (narrowly): 27.9% of fixes come from C (pred ~25%).
- P4 FAIL: only 3/24 WG breaks land on BASE's wrong vote answer (12.5%; pred ~45%, also FAIL). DPO does not import
  the vote's specific errors on test items.
- Exploratory (consistency.py): which items change is strongly consistent across the 3 seeds (over-dispersed vs
  independence in every stratum; e.g. C: 9 items fixed by all 3 seeds vs 1 expected, AG: 10 broken by all 3 vs 1).
  MIXR seeds disagree with each other on 203–242 of 1319 items, vs 7 for BASE run-to-run.

**Reading.** The net +51 items/seed (+3.8 pts) is the difference of large churn: +136 fixed vs −85 broken per seed.
Fixes are concentrated where BASE's own samples already found the answer (A+B: 72% of fixes, Lin et al.'s
"sharpening"), but C contributes as much net gain as B, so it is not only sharpening of a 4-sample-visible answer
(caveat: 4 samples is a weak probe). The main cost is breaking items that all 4 BASE samples got right (8.7%):
a seed-specific but systematic shift of the greedy path, not imitation of the vote's mistakes. Implication for
next steps: the gain could grow by cutting AG breaks (e.g. a KL/SFT anchor on own agreed-correct outputs, or
ensembling/merging seeds whose breaks differ) as much as by better labels. Seed-merging is testable on CPU only
if adapters were saved (they were not), so it belongs in a GPU job.
- Exploratory (ensemble.py): plurality vote of the 3 seeds' greedy answers = 75.97 vs seeds 73.1/72.6/75.0 and
  BASE G3 75.8 (4 generations). Seed diversity is usable, but at 3× inference it only matches BASE G3; a weight
  average (soup) of seed adapters would test whether it can be had at 1× cost (GPU, needs saved adapters).
