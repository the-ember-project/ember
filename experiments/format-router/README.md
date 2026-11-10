# format-router — label-free choice of reasoning format per task (EXPLORATORY)

Exploratory: post-hoc analysis of saved samples (gsm8k-mixvote, bbh-mixvote2, gridwalk2) to see whether within-format
sample agreement on a few calibration questions picks the better format, and whether that beats a fixed CoT+PAL mix
at equal sample count. Motivation: gridwalk2 and bbh-mixvote2 showed the best format is task-dependent. Any positive
finding must be re-tested with a written prediction on fresh data.

## Result (session #14, exploratory; `router.py` → `router-output.txt`)
Calibrate on the first 30 questions per task (label-free), evaluate on the rest (pooled n=1100 over 5 tasks).
- **Agreement-based task routing does not work.** It picked 5C on GSM8K (PAL agreement 0.65 < CoT 0.77, yet vote5P ≈
  vote5C and mixes are far better), and failed to pick PAL on BBH tracking_3 (agreement 0.53 vs 0.57, yet vote5P 57.3 vs
  vote5C 36.8). Correct picks only on logical_deduction, date (PAL mostly abstains) and gridwalk2. Pooled TASK-router(5)
  51.0% = vote(2C+3P) 51.0% (97/97), below MIX10 54.9% (−3.9, p=1e-4).
- **Per-question routing** (answer of the format whose samples agree more) is slightly *worse* than plain MIX10
  (−1.3, 5/19, p=0.007).
- Within-format agreement is not calibrated *across* formats: a format can agree less yet be as accurate (PAL on GSM8K
  makes varied arithmetic slips), or agree equally yet be far more accurate (PAL on tracking_3).
- MIX10 is the best or within ~3.6 pts of the best arm on every task; on GSM8K it beats both single formats by
  ~8 pts (true complementarity), elsewhere it hedges. Conclusion: keep the fixed mix; a router would need a signal
  other than raw agreement (e.g. abstention rate alone, or a learned verifier).
