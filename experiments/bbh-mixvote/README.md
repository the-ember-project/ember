# Does mixed-format voting (CoT + PAL) transfer to state tracking / spatial tasks? BBH, Qwen2.5-1.5B-Instruct (roadmap M2)

Why: on GSM8K (`../gsm8k-mixvote/`, confirmatory) a vote over 5 CoT + 5 PAL samples beat greedy CoT by +13.7 (p<0.001), apparently because the two formats make *different* wrong answers (same-wrong-answer rate 0.040 across formats vs 0.083 within CoT). M2 asks whether this carries to a second skill. Tasks are public BIG-Bench Hard (`lukaemon/bbh`, 250 test q each) that test state tracking and spatial reasoning (human's issue #2): `tracking_shuffled_objects_five_objects` (5-way multiple choice) and `navigate` (Yes/No, 58% "No").

## Design
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, HF transformers on a Kaggle T4 (not the Q4 llama.cpp setup used for GSM8K; all arms here share the same setup, so the comparisons are internal). Zero-shot.
- Per question: greedy CoT, 5 CoT samples (T=0.7), greedy PAL (program simulates the state and prints the answer), 5 PAL samples (T=0.7). Seed 777. max 512 new tokens. All 250 q of each task (n=500 pooled).
- Parsing: CoT → last `(X)` / last Yes|No; PAL → last printed line matched to the option text (accepted only if exactly one option matches), or Yes/No/True/False. Unparsed = wrong (None never wins a vote).
- Vote: plurality, ties → earliest candidate in order [CoT samples, PAL samples] (or [greedy CoT, ...] where stated). Same as GSM8K.

## Prediction (written before any BBH data from the 1.5B model exists; a 0.5B CPU smoke test on 3 q only checked parsing)
- **Primary H1 (confirmatory, = M2 criterion):** S2 = vote(5 CoT + 5 PAL samples) vs greedy CoT, pooled n=500. **Pass: diff ≥ +3 points AND exact two-sided sign-test p < 0.05.** Predicted **+6** (range 0 to +12); ~60% chance of pass.
- **H2 (mechanism):** binary answers force wrong answers to agree, so the gain on `navigate` should be smaller than on tracking. Predicted S2 − greedy: tracking **+10**, navigate **+2**. Pass: tracking gain > navigate gain.
- **H3:** PAL suits explicit state simulation better than GSM8K: greedy PAL ≥ greedy CoT + 3 on tracking (predicted +5). Opposite of GSM8K (−6.7).
- **H4:** S3 = maj5 CoT vs greedy, pooled: predicted +3; reported with p.
- Baselines predicted: greedy CoT tracking 30% (20–45), navigate 60% (50–70).
- Everything else (per-task p, other mixes, error correlation) is exploratory.

## Runs
GPU job: `run.py` (defaults) via `self/gpu_queue.txt`, limit 180 min. Output `gpu-output/*.jsonl`. Analysis: `python analyze.py` (written and smoke-tested before the run; applies the rules above).

## Results (confirmatory for H1–H4; Kaggle T4 run 2026-11-05, 108 GPU-min, Qwen2.5-1.5B-Instruct fp16, seed 777; `python analyze.py`)

| | tracking_5 (n=250) | navigate (n=250) | pooled (n=500) |
|---|---|---|---|
| greedy CoT | 18.4 | 61.2 | 39.8 |
| greedy PAL | 20.0 | 54.0 | 37.0 |
| maj5 CoT (S3) | 21.6 (+3.2, p=0.38) | 64.4 (+3.2, p=0.48) | 43.0 (+3.2, 89/73, p=0.24) |
| **vote 5 CoT+5 PAL (S2)** | 30.4 (+12.0, 55/25, p=0.001) | 68.0 (+6.8, 53/36, p=0.089) | **49.2 (+9.4, 108/61, p<0.001)** |
| [expl] maj5 PAL | 32.0 (+13.6, p=0.001) | 60.8 (−0.4) | 46.4 (+6.6, p=0.022) |
| [expl] pass@10 | 74.4 | 98.4 | 86.4 |

Tokens/q (pooled): greedy CoT 229; S2 2224 (9.7×).

- **H1 PASS** as pre-registered: pooled S2 +9.4, p<0.001 (predicted +6).
- **H2 PASS**: gain tracking +12.0 > navigate +6.8 (predicted +10 / +2; navigate gain larger than predicted).
- **H3 FAIL**: greedy PAL − greedy CoT on tracking = +1.6 (needed ≥+3; predicted +5).
- **H4**: S3 pooled +3.2 as predicted, but p=0.24: not distinguishable from noise.
- Baselines: tracking greedy 18.4% is below my predicted range (20–45) and at chance (5 options = 20%); navigate 61.2% in range.

### Scorer problem found after the run (affects the margin, not the verdict)
- `parse_cot` only accepted a letter in parentheses; the model often ends "So the answer is D." → 65/250 greedy tracking outputs unparsed (scored wrong). Re-parsing greedy CoT texts with a fixed parser (`answer is (X)` with optional parens): tracking 18.4 → **23.2%**, navigate unchanged.
- PAL programs often print part of an option ("black" for option "black ball"), which `parse_pal` rejects: 113/250 greedy PAL tracking outputs unmatched.
- Sample texts were **not saved** (only parsed answers), so S2 cannot be re-parsed. The same bugs also hit the samples, so S2 is handicapped too.
- **Conservative check** (S2 with bugged parser vs greedy with fixed parser): tracking +7.2 (52/34, p=0.066), navigate +6.8 (53/36, p=0.089), **pooled +7.0 (105/70, p=0.010)** — H1 still passes. Caveat that remains: voting over 10 samples is more robust to *format* failures than one greedy answer, so part of the gain may be format robustness rather than better reasoning. A rerun with fixed parsers and saved texts on fresh tasks (`experiments/bbh-mixvote2/`) addresses this.
- Interpretation: format-mixing transfers to non-arithmetic state tracking for this model (M2 met on the pre-registered criterion, with the scorer caveat above). Cost is high (9.7×); cheap mixes are untested on BBH.
