# Knowledge index

- `small-model-inference-tricks.md` — self-consistency, program-aided reasoning, verifiers: what to test and what was measured.
- `spatial-reasoning-benchmarks.md` — prior work for M2 (seqBench, Grid-Bench, ASCII drawing, walker delegation).
- Experiments: `experiments/gsm8k-baseline/` (greedy CoT baseline, `eval.py`), `experiments/gsm8k-selfconsistency/` (maj@5, `analyze.py` paired sign test), `experiments/gsm8k-pal/` (program-aided, `pal.py` with --k, `analyze.py`), `experiments/gsm8k-mixvote/` (CoT+PAL vote, offsets 300–599, confirmatory +13.7; `costcurve.py`, `errcorr.py`), `experiments/gsm8k-mixvote2/` (cheap 2 CoT+5 PAL mix, offsets 600–899, lab), `experiments/bbh-mixvote/` (M2: BBH tracking/navigate on Kaggle GPU, HF transformers `run.py`).
- Benchmarks: `experiments/benchmarks/README.md`.
- Roadmap: `research/roadmap.md`.
