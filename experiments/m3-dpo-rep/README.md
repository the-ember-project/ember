# m3-dpo-rep: does label-free MIXR-DPO's GSM8K *test* gain replicate on fresh seeds?

m3-stack run 3 (session #43) gave the first uncontaminated M3 hit: MIXR label-free DPO (chosen/rejected CoT picked by a
CoT+PAL vote, no gold labels) raised Qwen2.5-1.5B-Instruct greedy CoT on the **full GSM8K test set** from 69.7 to 73.1,
**+3.3 (124/80, p=0.0025)** — one training run. The playbook says one run per arm is not evidence (DPO and GRPO both
showed ~2-pt seed variance: m3-stack train-split +4.27 vs +2.2; GRPO MAJ +3.8/+3.0/+1.1). This job reruns the identical
recipe twice with fresh randomness.

## Design
- `run.py` = m3-stack/run.py with three changes: `--train_seed` (drives torch seed, pair subsampling and data order; was
  fixed 2029), per-replicate smoke dirs, and only the steps gen → train → test eval (BASE + MIXR greedy, 1319 items).
- Same 3000 GSM8K-train questions, same hyperparameters (k_cot 7, k_pal 4, T=0.8, cap 2000, β 0.1, α 1, LoRA r16, 1 epoch).
- Replicates: **R2** (vLLM seed 5051, train seed 3031), **R3** (6061, 3032); one per T4 in parallel (`rep.py`).
  R1 = m3-stack run 3 (4041, 2029), already measured.
- Each replicate re-evaluates BASE in its own process (greedy; should match m3-stack's 69.7 up to batching noise).
- Analysis: `python analyze.py` (paired sign tests per replicate vs its own BASE file).

## Prediction (written session #43, before any run)
- **D1 (each replicate):** MIXR − BASE on GSM8K test ≥ +2.0 and p < 0.05, reported separately for R2 and R3.
  Predicted: R2 +2.5, R3 +2.5 — each ~55% to pass.
- **D2 (M3 criterion, replicated):** mean gain over R1, R2, R3 ≥ +3.0 **and** at least 2 of the 3 individually p < 0.05.
  Predicted mean +2.8 → **FAIL narrowly** (~40%).
- **D3 (direction):** both R2 and R3 gains > 0. Predicted PASS (~85%).
- Validity: BASE test greedy in 67–72 in each replicate; each MIXR training ≥ 100 optimiser steps; MIXR parsed-answer
  rate ≥ 99%.
- Decision: D2 PASS → M3 counted as met on GSM8K (label-free DPO, uncontaminated test). D1 fails for both → the m3-stack
  +3.3 is reported as a single-seed result that did not replicate.

## Files
`rep.py` (driver), `run.py`, `analyze.py`; outputs in `gpu-output/R2/`, `gpu-output/R3/`.
CPU smoke: `python rep.py --cpu_smoke` (0.5B, 3 questions) — writes to `R2/`, `R3/` locally (deleted after the check).

## Results (confirmatory, analysed session #45; `python analyze.py`)
| Replicate | BASE | MIXR | diff | fixed/broken | p |
|---|---|---|---|---|---|
| R1 (m3-stack run 3) | 69.75 | 73.09 | +3.34 | 124/80 | 0.0025 |
| R2 (5051/3031) | 69.67 | 72.55 | +2.88 | 138/100 | 0.016 |
| R3 (6061/3032) | 69.67 | 74.98 | +5.31 | 146/76 | 3e-6 |

- Validity met for both new replicates (BASE in 67–72, 112 optimiser steps, 100% parsed). BASE agreement R1 vs R2 1312/1319.
- **D1 PASS** for R2 (+2.9, p=0.016) and R3 (+5.3, p=3e-6). Predicted +2.5 each.
- **D2 PASS: mean +3.84, 3/3 p<.05.** I predicted a narrow FAIL (+2.8); the result beat my prediction.
- **D3 PASS.**
- Decision (pre-registered): **M3 counted as met on GSM8K** — label-free MIXR-DPO (CoT+PAL vote picks chosen/rejected,
  no gold) raises Qwen2.5-1.5B-Instruct greedy CoT on the uncontaminated GSM8K test by +3.8 on average over 3 seeds
  (range +2.9 to +5.3). Seed spread ~2.4 pts, so any single-run comparison against this recipe needs ≥2 seeds.
