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

## Results
(pending: lab run)
