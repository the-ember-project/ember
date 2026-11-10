# m3-gridwalk: self-training on verified traces where a verifier exists (M3)

**Question.** On GSM8K, self-training was null (SFT, #8) or unstable (DPO, #9–#10). Gridwalk (10-step
relative-move state tracking; `gw.py` = copy of `experiments/gridwalk2/run.py`) is far below ceiling for
Qwen2.5-1.5B (greedy CoT ~39% in gridwalk2 interim, Q4) with *systematic* errors (turn semantics), and has an
exact simulator. Does rejection-sampling SFT (STaR/RFT style) on the model's own **verified** CoT traces raise
greedy CoT accuracy? And does a **label-free** filter — keep CoT traces that agree with the plurality of the
model's own *program* (PAL) samples — work as well, since PAL is much stronger than CoT on this task?

Prior work: STaR (Zelikman 2022), RFT (Yuan 2023) — SFT on own correct traces; ScPO / TTRL use self-consistency
labels. New here: cross-format (program-vote) pseudo-labels for CoT on a state-tracking task at 1.5B.

## Design (Kaggle T4, vLLM gen/eval, HF+PEFT train)
- Model `Qwen/Qwen2.5-1.5B-Instruct` fp16. One-shot chat demo prompts exactly as gridwalk2 (CoT and PAL).
- **Train tasks:** `make_task(i, 10, seed=9001)`, i < 1500. Per task 8 CoT + 5 PAL samples at T=0.8 (vLLM seed 5151).
- **Arms** (LoRA r=16 SFT on chosen CoT traces, prompt = system + demo + task, loss on the answer tokens only;
  lr 1e-4, 2 epochs, eff. batch 16; up to 2 distinct traces per task; equal size E = min(cap 1500, |GOLD|, |PALV|)):
  - **GOLD**: traces whose answer = simulator answer.
  - **PALV** (label-free): traces whose answer = plurality of the 5 PAL answers, if that plurality has ≥3 votes.
  - **COTM** (label-free control, ScPO-like): traces whose answer = plurality of the 8 CoT answers with ≥4 votes;
    size min(E, available) — reported, smaller E is noted.
- **Eval (greedy CoT, fresh tasks never used in training):** primary 600 tasks `make_task(i, 10, seed=9002)`;
  secondary 300 tasks at 14 steps (seed 9003, length transfer); drift 300 GSM8K test questions (greedy CoT).
- Scoring: `gw.parse_cot` vs simulator; paired sign test (exact). Verdicts from per-item `eval_*.jsonl`.

## Prediction
(written before any run; seeds above are fresh)
- **Validity:** BASE greedy skip rate (answers <30 tokens) ≤20% on the primary eval; BASE accuracy expected 30–50%.
- **H1:** GOLD − BASE ≥ +3 points with sign-test p < 0.05 on the 600 primary tasks. Predicted **+10** (~70% chance).
- **H2:** PALV − BASE ≥ +3, p < 0.05. Predicted **+8** (~60%).
- **H3:** PALV label precision (label = simulator answer, on tasks where PALV gives a label) ≥ 90%. Predicted 93%.
- **H4:** PALV − COTM ≥ +3, p < 0.05. Predicted +5 (~40%; COTM labels are expected to be much noisier, CoT sample acc ~25%).
- Secondary (reported, no pass/fail): 14-step transfer for each arm; GSM8K drift (tolerance: each arm within −3 of BASE).
- If H1 passes, M3 counts as met *for a task with an exact verifier*; H2 would make it label-free.

## Results (Kaggle 2026-11-10, 206 GPU min; CONFIRMATORY; verdicts recomputed from eval_*.jsonl in session #15)
Train-set sizes: E=1013 traces per arm (GOLD 1500 labelled tasks/1830 traces available; PALV 814 tasks labelled, 1013 traces; COTM 739 tasks, 1403 traces → capped at 1013).
Label precision: GOLD 100%, **PALV 86.9%** (traces whose answer is truly correct 95.2%), COTM 61.4%.

| greedy CoT acc | gw10 (600, primary) | gw14 (300, transfer) | GSM8K (300, drift) |
|---|---|---|---|
| BASE | 46.3 | 20.0 | 69.7 |
| GOLD | 70.2 (+23.8, 156/13, p<1e-30) | 43.0 (+23.0, 75/6) | 68.3 (−1.3, p=0.54) |
| PALV | 63.2 (+16.8, 112/11, p<1e-20) | 36.7 (+16.7, 56/6) | 70.0 (+0.3) |
| COTM | 47.3 (+1.0, 33/27, p=0.52) | 23.0 (+3.0, p=0.15) | 68.3 (−1.3) |

- **Validity PASS:** BASE skip rate (<30 tokens) 0.0%; BASE acc 46.3% (predicted 30–50).
- **H1 PASS:** GOLD − BASE +23.8, p<1e-30 (predicted +10).
- **H2 PASS:** PALV − BASE +16.8, p<1e-20 (predicted +8). Label-free.
- **H3 FAIL:** PALV label precision 86.9% < 90% (predicted 93%).
- **H4 PASS:** PALV − COTM +15.8 (116/21), p<1e-16 (predicted +5).
- Secondary: gains carry to 14-step tasks (longer than any training task) at the same size; GSM8K drift within −3 for all arms.
- GOLD − PALV = +7.0 (56/14, p<0.001): label noise (13% wrong labels, fewer tasks covered) costs about 30% of the gold gain.

**Interpretation.** M3 met on a task with an exact verifier, and the label-free version works too: the model's own *program* votes are a good enough teacher for its *CoT*. Same-format majority (COTM, ScPO-like) does nothing because its labels are 61% correct: they reinforce the systematic errors. The trained greedy CoT (GOLD 70.2%, fp16) reaches about the level of the base model's 10-sample vote (gridwalk2 MIX10 69.5%, Q4, different tasks; not a paired comparison) at 1× cost. Caveats: one training run per arm (but effects are 5–8× the DPO run-to-run spread); a synthetic own task.
