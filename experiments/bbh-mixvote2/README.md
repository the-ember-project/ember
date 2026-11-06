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

## Results (confirmatory for H1–H4; analysed session #6, 2026-11-06)
Kaggle T4, 123 GPU min. `python analyze.py` (full output reproducible from gpu-output/).

| Pooled n=750 | acc | diff vs greedy CoT | fixed/broken | p | tokens × greedy |
|---|---|---|---|---|---|
| greedy CoT | 35.7 | — | — | — | 1.0 |
| **S2 vote(5 CoT+5 PAL)** | 42.5 | **+6.8** | 162/111 | **0.002** | 8.7 |
| S3 maj5 CoT | 37.3 | +1.6 | 139/127 | 0.50 | 5.6 |
| C vote(1 CoT+3 PAL) | 35.1 | −0.7 | 149/154 | 0.82 | 3.0 |
| greedy PAL | 21.5 | −14.3 | 102/209 | <0.001 | 0.5 |

- **H1 PASS** (+6.8, p=0.002; predicted +6).
- **H2 PASS** S2 vs S3: +5.2, 54/15, p<0.001 (predicted +4).
- **H3 FAIL** cheap mix: −0.7, p=0.82 (predicted +4), cost 3.0×. Per task: tracking_3 +15.6 (p=0.001), logical_deduction −6.0 (p=0.12), date −11.6 (p=0.005).
- **H4 PASS (on the ≥0 criterion)**: logical_deduction S2 +3.2, p=0.43 (predicted +2); not distinguishable from 0.
- **Scorer check:** greedy-CoT unparsed 2.0% / 4.8% / 5.2% — date slightly over 5%. Hand-check of date unparsed: truncated/looping generations, not parser misses, so it's a model failure, not a scorer bug.
- Baselines: tracking_3 34.8 (pred 40), logical_ded_5 27.2 (pred 35), date 45.2 (pred 50); all inside predicted ranges, all below prediction.

Per-task S2: tracking_3 +13.2 (p=0.002), logical_ded +3.2 (p=0.43), date +4.0 (p=0.33).

**Interpretation.** PAL fits badly on logical_deduction and date: greedy PAL prints something that matches no option on 72% / 93% of questions (e.g. "13/24", "kiwi"); checked, these are genuine program failures, not parser misses. An unmatched PAL output abstains from the vote. So the big 5 CoT+5 PAL vote stays safe: when PAL abstains, the 5 CoT samples decide. The cheap PAL-heavy mix does not: it falls back to a single T=0.7 CoT sample, which is worse than greedy. **Lesson: the PAL-heavy cheap mix (the M1 GSM8K winner) is only safe where PAL fits the task. Either keep ≥3 CoT samples, or route by PAL parse rate.** Exploratory: vote(greedy CoT + greedy PAL + 5 CoT) +7.2, 99/45, p<0.001, at ~7×. On tracking_3, maj5 PAL alone gives +24.4, so where PAL fits it dominates.

Supersedes the scorer caveat in bbh-mixvote: with fixed parsers and fresh tasks, the M2 gain replicates (+6.8, p=0.002).
