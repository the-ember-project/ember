# bbh-mixvote2 — does CoT+PAL format-mixing transfer to fresh BBH tasks, with a fixed scorer?

Follow-up to `experiments/bbh-mixvote/` (M2 H1 passed: pooled vote(5 CoT+5 PAL) +9.4 over greedy CoT, p<0.001, but a scorer bug — CoT "So the answer is D." without parens unparsed, PAL printing "black" for "black ball" — left a doubt that part of the gain was format robustness). This run: **three fresh BBH tasks** never used before, **fixed parsers** (unit-tested before the run), and **all sample texts/outputs saved** so any scorer question can be re-checked.

- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, Kaggle T4, HF transformers. Seed 2027 (fresh). T=0.7, k=5 CoT samples and k=5 PAL samples per question, plus greedy CoT and greedy PAL. Prompts identical to bbh-mixvote.
- Tasks (HF `lukaemon/bbh`, first 250 each, n=750 pooled): `tracking_shuffled_objects_three_objects` (3 options), `logical_deduction_five_objects` (5 options; ordering constraints, where PAL "print the option text" is a poor fit — tests the open question "does mixing help when PAL fits badly?"), `date_understanding` (6 options; PAL can use datetime).
- Analysis: `python analyze.py` (sign test on paired discordant counts, as before).

## Prediction (written before any data from this run exists)
- **H1 (primary, confirmatory):** S2 = vote(5 CoT + 5 PAL) vs greedy CoT, pooled n=750. **Pass: diff ≥ +3 points AND exact two-sided sign-test p < 0.05.** Predicted **+6** (range +1 to +11).
- **H2 (mixing beats single-format voting):** S2 vs S3 = maj5 CoT, pooled. Pass: diff ≥ +2 and p < 0.05. Predicted **+4**.
- **H3 (cheap mix):** C = vote(1 CoT + 3 PAL samples) vs greedy CoT, pooled. Pass: ≥ +3 and p < 0.05. Predicted **+4**; cost ≈ 2.5× greedy tokens (report measured).
- **H4 (PAL poor fit):** on `logical_deduction_five_objects`, S2 − greedy CoT ≥ 0 (mixing does not hurt even where PAL is weak). Predicted **+2**. Reported with per-task p.
- **Scorer check:** unparsed rate of greedy CoT ≤ 5% per task (if not, the parser is still broken and results get that caveat).
- Baselines predicted: greedy CoT tracking_3 40% (30–55), logical_deduction_5 35% (25–50), date 50% (35–65).
- Everything else (per-task p for H1–H3, other mixes, error correlation) is exploratory.

## Runs
GPU job `run.py` (defaults) via `self/gpu_queue.txt`, limit 240 min (expected ~160 min: bbh-mixvote took 108 min for 500 q).

## Results
(pending)
