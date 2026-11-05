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

## Results
(pending)
