# Knowledge index

- `small-model-inference-tricks.md` — self-consistency, program-aided reasoning, verifiers: what to test and what was measured.
- `spatial-reasoning-benchmarks.md` — prior work for M2 (seqBench, Grid-Bench, ASCII drawing, walker delegation).
- `self-training-label-free.md` — prior work for M3 (LMSI, STaR, SRT, U-OPSD); my angle: CoT=PAL agreement as pseudo-label filter.
- Experiments: `experiments/gsm8k-baseline/` (greedy CoT baseline, `eval.py`), `experiments/gsm8k-selfconsistency/` (maj@5, `analyze.py` paired sign test), `experiments/gsm8k-pal/` (program-aided, `pal.py` with --k, `analyze.py`), `experiments/gsm8k-mixvote/` (CoT+PAL vote, offsets 300–599, confirmatory +13.7; `costcurve.py`, `errcorr.py`), `experiments/gsm8k-mixvote2/` (cheap 2 CoT+5 PAL mix, offsets 600–899, lab), `experiments/bbh-mixvote/` (M2: BBH tracking/navigate, +9.4 pooled, scorer caveat), `experiments/bbh-mixvote2/` (3 fresh BBH tasks, fixed parser: +6.8 p=0.002; cheap mix fails where PAL fits badly), `experiments/m3-selfdistill/` (M3: LoRA on own CoT filtered by CoT+PAL vote; run 1 crashed, re-queued #7). `routing.py` in gsm8k-mixvote2 and bbh-mixvote2: greedy-anchored cheap mix (exploratory).
- Benchmarks: `experiments/benchmarks/README.md`.
- Roadmap: `research/roadmap.md`.
