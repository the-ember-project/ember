# m3-grpo-rep — does label-free GRPO (majority reward) replicate?

**Why:** m3-grpo-main (run 3, confirmatory) gave MAJ − BASE **+3.8** (104/66, p=0.004, H2 PASS) on MATH test
positions 2700–3699, but GOLD − BASE only +1.6 (H1 FAIL) and GOLD − MAJ −2.2 (p=0.09). A gold reward doing worse than
a ~90%-accurate majority reward suggests run-to-run training variance of ~2 points (the playbook already records that one
training run per arm is not evidence: DPO gave +3.1 and −7.9 on two data draws). So the H2 pass may be seed luck. This
is the replication that decides whether M3 (label-free training raises greedy accuracy ≥3 pts on held-out items) is met
on MATH.

**Design (identical recipe to m3-grpo-main run 3 except seeds, prompt draw and eval slice):** Qwen2.5-1.5B-Instruct,
LoRA r16, GRPO (TRL 1.18.1, vLLM colocate), 100 steps × 4 prompts × 8 completions, 1024 completion tokens, lr 3e-5,
beta 0, reward = 1 if the boxed answer equals the group's majority answer (no gold). Two independent replicates:
MAJ2 (train seed 8, MATH-train prompt shuffle seed 32), MAJ3 (seed 9, shuffle 33) — run 3 used seed 7 / shuffle 31.
Eval: greedy, 2048 tokens, MATH test positions **0–999** of the seed-2029 shuffle (numeric golds; never used for any
GRPO run; only base-model inference/DPO evals of other experiments touched them — this model never trained on them),
plus BBH date (n=250) drift. Same scorer as m3-grpo-main (`analyze.py`, `rep_analyze.py`).

## Prediction
Paired two-sided exact sign test per replicate on the 1000 items. Confirmatory.
- **R1 (each replicate):** MAJk − BASE ≥ +2.0 and p < 0.05, reported separately for MAJ2 and MAJ3.
  Predicted +2.5 each (~50% chance each passes).
- **R2 (M3 criterion):** mean of (MAJ2 − BASE, MAJ3 − BASE) ≥ +3.0 **and** both replicates individually p < 0.05.
  Predicted mean +2.5 → FAIL (I expect the run-3 +3.8 to have been on the lucky side).
- **R3 (not only finishing/boxing):** mean replicate gain on items where BASE produced a \boxed answer ≥ +1.5. Predicted +2.0.
- Drift (descriptive): BBH-date MAJk − BASE ≥ −4.
- **Validity** (else compromised): BASE accuracy 50–62; BASE boxed ≥ 88%; each replicate ≥ 90 training steps; 1000 items per arm.
- Decision: R2 PASS → M3 counted as met on MATH (label-free GRPO), and next try the ORM-reward arm. R1 passes for
  neither → run 3's +3.8 was luck; label-free RL at this budget gives ≤ ~2 pts; write up M3 as not met with RL.
  Mixed → effect real but < 3 pts and noisy.

## Files
- `run.py` (Kaggle driver, from m3-grpo-main with `--shuf`, replicate arm names, eval offset 0), `rep_analyze.py` (verdicts).
