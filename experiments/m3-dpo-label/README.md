# m3-dpo-label: is label quality or teacher content the bottleneck for 1.5B DPO on GSM8K?

Session #46 (2026-11-26). Builds on m3-stack / m3-dpo-rep (label-free MIXR-DPO, +3.8 mean over 3 seeds on the
full GSM8K test). Working hypothesis (agenda): label-free training helps when the label source is a stronger solver
than the policy. MIXR's label precision on pair-questions is already ~0.93, so better *labels* may not help; what the
1.5B lacks may be better *solutions* to imitate. This job separates the two and is the first M5 step (match 3B).

## Design
Same 3000 GSM8K-train questions (shuffle seed 2029, first 3000; as m3-dpo-rep). One Kaggle job, 2 T4s.
- 1.5B generation (GPU0, seed 9091): greedy + 7 CoT (T=0.8), greedy + 4 PAL — exactly the MIXR recipe.
- 3B teacher generation (GPU1, seed 9092): Qwen2.5-3B-Instruct greedy + 4 CoT, greedy + 1 PAL, same prompts.
- Arms (DPO + NLL on chosen, LoRA r16, β 0.1, α 1, lr 5e-5, 1 epoch — unchanged recipe; equal pair count E for all arms):
  - **MIXR**: chosen/rejected = 1.5B's own CoT, label = 1.5B CoT+PAL vote (label-free; 4th replicate).
  - **GOLD**: same pair construction with the gold answer as label (label-quality ceiling; uses gold).
  - **T3D**: chosen = 3B CoT samples agreeing with the 3B label (plurality ≥4 of its 7 answers, no gold);
    rejected = 1.5B CoT samples disagreeing with that label (teacher distillation with on-policy negatives; no gold).
- Train seeds: GPU0 trains MIXR, T3D (seed 4051); GPU1 trains GOLD, T3D (seed 4052) → T3D has 2 seeds.
- Eval: greedy CoT on the full GSM8K test (n=1319, never trained on), max_new 512. 3B greedy reference = m3-stack
  `test3b_eval_BASE.jsonl` (85.2, same items, prompt and cap; 275 tok/q, so the cap rarely binds).
- Driver `lab.py` runs the whole pipeline on a 16-question slice first. Analysis: `python analyze.py`.

Prior work: Li et al. 2502.12143 (verified abstract): ≤3B students learn poorly from *much* stronger/longer-CoT
teachers; smaller teachers or mixing helps. A 3B same-family teacher is a small gap, so it should be learnable.

## Prediction (written session #46, before any run)
All comparisons paired on the full GSM8K test, exact sign test.
- **L1 (label-quality ceiling):** GOLD − MIXR ≥ +2.0 and p < .05. Predicted **+1.0 → FAIL** (~25% pass).
- **L2 (teacher content beats self-labels):** mean over the 2 T3D seeds of T3D − MIXR ≥ +2.0, both seeds > 0.
  Predicted **+1.0 → FAIL** (~35% pass).
- **L3 (distillation works at all):** each T3D seed − BASE ≥ +3.0 with p < .05. Predicted **PASS** (≈ +4.5).
- **L4 (M5 check):** best T3D seed ≥ 3B greedy − 2 (≥ 83.2). Predicted ~74.5 → **FAIL**.
- Secondary: MIXR − BASE predicted +3.5 (4th seed); GOLD − BASE predicted +4.5.
- Validity: BASE 67–72; parsed-answer rate ≥ 99% for every arm; E ≥ 1000 pairs; report 3B greedy train accuracy and
  T3D label precision on pair-questions (expected ≥ 0.9). One seed for MIXR/GOLD: a gap < ~2.5 is within seed spread.
- *Addendum (#46, still before any run; criteria unchanged):* after writing the above I re-read m3-dpo (older recipe,
  full test): GOLD +5.5 vs COT-majority +3.1 vs strict MIX +2.35, i.e. GOLD beat label-free by ~2.4 there, which I
  attributed to GOLD covering more hard questions (34.6% vs 15.5%). That raises my L1 pass chance to ~40% (predicted
  GOLD − MIXR ≈ +1.5). T3D also covers questions the student never solves, so the same reading favours T3D.
- Reading: L1 FAIL & L2 PASS → content, not labels, is the bottleneck → scale teacher distillation (M5 route).
  L1 PASS → labels matter → try a stronger label source (3B vote) on own samples. Both FAIL → the DPO recipe saturates
  around +4 whatever the signal; look elsewhere (data volume, iteration: m3-dpo-iter).

## Status
- #46: CPU smoke of the whole pipeline (0.5B as both models, faked labels via ITER_FAKE_LABELS) ran clean, analyze.py
  ran on its output; build_pairs unit-tested on synthetic rows (T3D chosen from teacher texts, rejected from student;
  T3D also covers questions where every student sample is wrong). Queued on Kaggle `lab.py 220` behind m3-dpo-iter.
  Estimate ~185 GPU min (3B gen ~95 min is the long pole).
