# Program-aided solving (PAL) vs chain-of-thought on GSM8K, Qwen2.5-1.5B-Instruct Q4_K_M

The model writes a short Python program (zero-shot, greedy, one ```python block, `print(answer)`); `pal.py` runs it in a subprocess (10 s timeout) and takes the last printed number. Same question sampling as `../gsm8k-baseline/eval.py` (seed 1234 shuffle), so every question is paired with an existing greedy-CoT answer. Idea: Gao et al. 2022 (PAL), Chen et al. 2022 (Program of Thoughts): moving arithmetic to the interpreter removes calculation slips.

Runs: offset 0, n=100 (session #2, the baseline set) and offset 100, n=200 (lab, fresh questions that already have greedy-CoT answers in `../gsm8k-baseline/results_qwen1.5b_greedy_off100_n200.jsonl`).

## Prediction (written before running)
- Hypothesis: for this 1.5B model, a sizeable share of CoT errors are arithmetic, so PAL beats greedy CoT at similar or lower token cost.
- Primary metric: accuracy on all 300 paired questions (offset 0–299), PAL vs greedy CoT, both scored by exact numeric match (raw scorers, no hand correction on either arm). Report discordant pairs (fixed/broken) and an exact sign-test p-value.
- Predicted: PAL = greedy CoT **+2 to +10 points** (CoT on the 300 is 68.3%); program execution error rate < 10%; tokens per question below CoT's ~234.
- Pass criterion: PAL ≥ CoT + 3 points on the 300 paired questions. A result inside ±3 is inconclusive; PAL ≤ CoT − 3 is a negative result.
- Secondary (exploratory, no pass criterion): (a) hybrid = PAL answer, falling back to the CoT answer when the program errors; (b) agreement: accuracy when PAL and CoT agree, and how often they agree — a cheap two-method vote/verifier signal.

## Results
### Partial: offset 0, n=100 (session #2, session sandbox) — first third of the pre-registered 300; the pass/fail verdict waits for the lab run
`python experiments/gsm8k-pal/analyze.py`:
- PAL **64%** vs CoT 68% (−4 pts), 11 fixed / 15 broken, sign test p=0.56.
- Execution errors: 2% (1 timeout, 1 no output), so the predicted <10% was met. Tokens: 126/q vs CoT 234 (prediction met: cheaper). Speed ~6.5 s/q in the sandbox.
- Exploratory: hybrid (fallback to CoT on error) 65%. PAL and CoT agree on 59/100 questions and are **89.8% correct when they agree**. Either one correct: 79%.
- Exploratory: the 15 broken cases are almost all wrong *program logic* (wrong setup, ×2 / sign / rate errors), not crashes. For this 1.5B model, arithmetic slips are a smaller share of CoT errors than I assumed. PAL's value looks like a *second, partly independent opinion* rather than a better solver.

### Final: offsets 0–299, n=300 paired (session #3; offset 100–299 ran in the lab) — **confirmatory, negative**
`python experiments/gsm8k-pal/analyze.py`:
- PAL **61.7%** vs greedy CoT 68.3% (**−6.7 pts**), 26 fixed / 46 broken, exact sign test **p = 0.024**. Pre-registered pass (≥ +3) **not met**; by the pre-registered rule (≤ −3) this is a **negative result**: zero-shot PAL is significantly worse than CoT for Qwen2.5-1.5B Q4 on GSM8K. Predicted +2 to +10 — prediction wrong.
- Execution errors/no output: 4.0% (prediction < 10%: met). Tokens/q: 130 vs 233 (prediction "cheaper": met).
- Exploratory: hybrid (CoT fallback on error) 63.3%. Agreement: PAL = CoT on 169/300, **94.1% correct when they agree**; either correct 77.0%.
- Interpretation (exploratory): for this model, writing a program is a harder format than prose CoT (wrong setup/logic dominates), so the tool doesn't pay for itself as a solver. Agreement remains a strong confidence signal → tested as a voter in `../gsm8k-mixvote/`.
- Rerun note: `pal.py` gained `--k/--temp/--sample_seed` in session #3; the k=1 greedy path is unchanged.
