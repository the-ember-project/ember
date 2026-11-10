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

## Results
(pending — queued on Kaggle after m3-dpo3)
