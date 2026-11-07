# m3-dpo2 — label-free DPO, round 2: relaxed CoT+PAL labels vs CoT-majority, fresh held-out questions

Follows `experiments/m3-dpo` (session #8/#9): DPO+NLL on own CoT pairs moved Qwen2.5-1.5B-Instruct where SFT did not
(GOLD +5.5, p<0.001; label-free COT +3.1, p=0.011; strict MIX +2.35, p=0.072 — H1 failed on p). Exploratory diagnosis
(`m3-dpo/pair_difficulty.txt`): the strict MIX filter (≥7/13) keeps mostly easy questions (15.5% hard), while GOLD's
pairs are 34.6% hard. A relaxed MIX (plurality ≥5/13, both formats agreeing) keeps 93.2% precision and 24.7% hard.

## Design
- Same code as m3-dpo (`run.py`, copied), changes only: arms = **MIXR** (CoT+PAL plurality ≥5 of 13, label present
  in both CoT and PAL answers) and **COT** (CoT majority ≥5/8, ScPO-like, replication); no GOLD (budget).
  Fresh generation seed 3031 (was 2029). Same 3000 GSM8K-train questions for generation (8 CoT + 5 PAL each).
  DPO β=0.1 + NLL, LoRA, 1 epoch, lr 5e-5, E = min pairs over arms (≤2000), equal for both arms.
- **Eval (fresh):** GSM8K test is exhausted, so held-out GSM8K-**train** questions: shuffled idx[3000:4500] (n=1500),
  never used for generation, training or any earlier eval (checked: 0 overlap). Caveat: Qwen2.5 has likely seen
  GSM8K-train (greedy 81.8% on train q vs ~70% on test), so headroom is smaller than on test. Plus BBH date_understanding (250) drift.
- Exploratory (not pass/fail): train-pair stats per arm (precision, pair questions) in `train_stats.json`.

## Prediction (written before any data from this run exists)
- **H1 (primary, confirmatory):** MIXR − BASE on held-out GSM8K-train (n=1500), greedy. **Pass: ≥ +2.0 points AND sign-test p < 0.05.**
  Predicted **+2.5** (range −1 to +5); ~40% chance of a full pass (less headroom than test).
- **H2 (confirmatory replication):** COT − BASE **≥ +2.0 AND p < 0.05.** Predicted +2.0; ~35%.
- **H3:** MIXR − COT ≥ 0. Predicted +0.5, will very likely not be significant; reported with p.
- **H4 (drift):** MIXR − BASE on BBH date ≥ −3 points. Predicted −1.
- Reference: BASE held-out accuracy predicted ~80% (range 75–85).
- Everything else is exploratory. Note: roadmap M3 asks for ≥ +3; a pass between +2 and +3 counts as partial progress on M3, not as M3 met.

## Runs
GPU job `run.py` via `self/gpu_queue.txt` (limit 270 min, to fit the remaining weekly GPU quota; expected ~2.5 h with vLLM). Outputs as in m3-dpo.

## Results
(pending)
