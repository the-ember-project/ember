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

## Results (Kaggle, 144 GPU min, 2026-11-08; confirmatory)
**Bug in summary.json:** the summary loop still used the m3-dpo arm name `MIX`, so `MIXR-BASE` and `MIXR-COT` were never written. All paired stats below were recomputed in session #10 from the saved per-question files `gpu-output/eval_*.jsonl` (same sign test). COT-BASE matches summary.json exactly.

Accuracy, greedy, held-out GSM8K-train idx[3000:4500] (n=1500): BASE 80.5%, **MIXR 84.7%**, COT 72.7%. BBH date (n=250): BASE 50.0, MIXR 45.2, COT 42.0.

| Hypothesis | Measured | Verdict |
|---|---|---|
| H1 MIXR−BASE ≥+2, p<0.05 | **+4.2**, 142 fixed / 79 broken, p<0.001 | **PASS** (predicted +2.5) |
| H2 COT−BASE ≥+2, p<0.05 | **−7.9**, 116/234, p<0.001 (significant harm) | **FAIL** |
| H3 MIXR ≥ COT | +12.1, 247/66, p<0.001 | PASS |
| H4 MIXR drift on BBH date ≥−3 | **−4.8**, 31/43, p=0.20 | **FAIL** (not significant, but below the tolerance I set) |
| BASE ~80% | 80.5% | as predicted |

Training: both arms E=1799 pairs, 113 steps. Final DPO margin MIXR 2.2 (pair acc 0.63), COT 5.0 (0.76). Pair labels: MIXR precision 93.0% on 1313 pair questions (chosen correct 92.0%); COT 89.7% on 1190 (chosen correct 88.2%). COT outputs got ~9% longer (mean 791 vs 725 chars); hand-check of 3 COT-broken items: genuine reasoning errors (extra multiplications, wrong subtraction), not format/parse failures.

**Interpretation (exploratory).** The pre-registered label-free cross-format arm passed with +4.2 — above the M3 bar of +3 — on held-out questions. But the same recipe with CoT-majority labels, which gave +3.1 on GSM8K test in m3-dpo, here *lost* 7.9 points. The only differences from m3-dpo for COT are the generation seed (3031), the eval set, and E. So label-free DPO is **unstable across data samples**: one run per arm is not enough, and the MIXR win may partly be luck of the draw too. The COT arm's training stats looked like m3-dpo's (margin ~5), so training curves do not flag the collapse. Possible mechanism (untested): COT's 12% wrong chosen answers plus its larger margin push harder in wrong directions; MIXR's lower margin suggests its pairs are harder to separate (more conflicting signal = implicit regularisation?). Drift on BBH date is negative for every DPO arm in both runs (−2 to −8), so some general-skill cost is likely real.
**Next:** replicate MIXR (and COT) with 2 more generation seeds, smaller eval; add an NLL-only control (reviewer); track drift on a second non-GSM8K task.

