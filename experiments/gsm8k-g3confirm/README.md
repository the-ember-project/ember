# gsm8k-g3confirm: fresh confirmation of the greedy-anchored cheap mix (G3)

Session #7 found G3 post hoc (exploratory): vote(greedy CoT + 3 PAL samples), with greedy winning ties. It scored +9.3 on GSM8K 600–899 (2.7× tokens) and +4.9 on BBH, and never fell below greedy on any task. This run tests it on questions it was not found on.

## Design
- Qwen2.5-1.5B-Instruct Q4_K_M, llama-cpp, CPU lab. Prompts and scorers: gsm8k-baseline/eval.py (greedy CoT) and gsm8k-pal/pal.py (PAL, k=4, T=0.7, **fresh sample_seed 5151**).
- GSM8K test offsets **900–1318** (`random.Random(1234)` shuffle, n=419, the rest of the test set). For this Q4 model these questions are fresh. The fp16 model's m3-selfdistill run 2 evaluated them, but G3 was not selected on them.
- Vote: answers rounded to 4 dp, `None` skipped, ties go to the first answer seen (greedy). Paired exact sign test vs greedy. Cost = tokens relative to greedy.

## Prediction (written before running)
- **Primary:** G3 − greedy ≥ **+5 points** AND sign-test **p < 0.05**, at cost ≤ 3.5×. Predicted **+8** (range +4 to +12); greedy predicted ~65% (range 58–72).
- Secondary (reported, not pass/fail): G4 ≥ G3; G2 > greedy. vote(4PAL) alone vs greedy is only a reference.
- Everything else is exploratory.

## Results (session #9; confirmatory; CPU lab, 211 min)
From `analysis.txt` (n=419, offsets 900–1318, fresh PAL seed 5151):
- greedy CoT 66.8% (predicted ~65, in range).
- **Primary PASS:** G3 vote(greedy+3 PAL, greedy wins ties) 73.7%, **+6.9**, 41 fixed / 12 broken, p=0.0001, cost 2.65× (≤3.5×). Predicted +8 — a bit below but inside range.
- Secondary: G4 +6.4 (46/19, p=0.001, 3.20×) — G4 ≥ G3 **not** met (−0.5, noise). G2 +4.5 (29/10, p=0.003, 2.10×) > greedy: met.
- Reference: vote(4 PAL) without the greedy anchor +1.4, p=0.62 — the anchor/mix carries the gain, not PAL alone.
- Conclusion: the cheap greedy-anchored mix found post hoc in session #7 replicates on fresh GSM8K data: ~+7 points at ~2.7× tokens (vs +11.7 at 4.75× for vote(2 CoT+5 PAL)). GSM8K test is now exhausted (offsets 0–1318 all used).
