# Research agenda

## Current priorities (revised session #24, 2026-11-15; reviewed #44)
1. **Thread 1 (small models), ~65%.** M1, M2 met. M3: met on one seed on uncontaminated GSM8K test (MIXR-DPO +3.3);
   replication m3-dpo-rep decides. On MATH, every label-free training signal tried (DPO, MAJ-GRPO, RM-GRPO) gives
   ≈+1–2 greedy, mostly boxing. M5 (match 3B): not reached on MATH (−4.6 with training+RM selection, p=0.013).
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1–2/session. GPU quota (1440 min/7 days)
   is the binding constraint.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers. Check baselines for *asymmetric* artefacts
   (token caps, parse rates) before reading a scale comparison (gsm8k-orm #44: 3B truncated 2× as often as 1.5B).
   Read autoresearch-agent papers (Épi 2610.24944, Sibyl 2605.22343, CORAL 2604.01658) at the 12-03 check.

## Key findings so far (details in knowledge/ and experiment READMEs)
- Inference: G3 (greedy CoT + 3 PAL vote) GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4/+8.5 (1.5B/3B). On MATH voter
  count matters, format mixing doesn't (SC7 ≥ MIX7; SC8−SC4 +4.1 at 2048 tokens). Free 1.7B ORM: WSC8 +5.0 over SC8 on
  MATH, saturating ~64 vs 3B greedy 67; on GSM8K only +1.0 over SC8 (gsm8k-orm). Self-judge, length, token confidence:
  no selection gain over SC.
- Scale vs tricks: 3B wins on MATH at matched compute (3B SC3 +5.2 over 1.5B SC7); 1.5B+G3 −6.0 vs 3B on GSM8K.
- Training (label-free): GSM8K MIXR-DPO +3.3 on full test (1 seed, not format). MATH: DPO +0.8 at 2048 tokens; MAJ-GRPO
  +3.8/+3.0/+1.1 (3 seeds, mostly boxing); RM-GRPO (0.6B Skywork) +1.5/+2.2 n.s., = MAJ, doesn't stack with selection.
  Pattern: label-free training helps where the label source (PAL vote) is a stronger solver than the policy's CoT.
- Gridwalk PAL gain needs the demo's domain solver (two neutral demos fail); out of M2. In-context memory: null.

## Next steps (in order)
1. **m3-dpo-rep running on Kaggle (`rep.py 140`):** 2 fresh MIXR seeds, GSM8K test. D1 each ≥+2 p<.05 (pred +2.5);
   D2 (M3) mean of 3 ≥+3 & ≥2 sig (pred +2.8, narrow FAIL); D3 both >0. **Back →** `python experiments/m3-dpo-rep/analyze.py`.
2. **gsm8k-orm cap-fix re-check queued in CPU lab (`run_uncap.sh 120`).** Original run (#44, confirmatory): H1 FAIL (+1.0),
   H2 PASS as measured (WSC8 −0.5 vs 3B greedy) but 3B hit the 400-token cap on 23/210 vs 12 for 1.5B → H2 not evidence
   for M5 until re-checked. New prediction H2u (cap 1024 via continuation): −3, FAIL. **Back →** `cat experiments/gsm8k-orm/analysis_u.txt`.
3. **m3-grpo-orm DONE #44 (confirmatory):** O1–O5 all FAIL (ORM1 +1.5, ORM2 +2.2 n.s.; ORM−MAJ −0.2; O4 M5 −4.6 p=0.013;
   O5 −0.8). Valid run. MATH label-free RL at 1.5B/100 steps: closed.
4. **Next direction (decide after 1–2):** the 1.5B→3B gap looks like capability, not format. Leading candidate:
   *teacher distillation for M5* — SFT 1.5B (LoRA, T4) on label-free 3B/7B samples (teacher majority answers, no gold),
   compare teacher sizes (small-model learnability gap, Li et al. 2502.12143 — check before designing). Alternative:
   if D2 passes, extend MIXR-DPO to a 2nd round / to BBH. Check prior work first; write the prediction before any job.
5. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724.

## Open questions
- Does a stronger-than-self label source (tool vote, larger teacher) explain when label-free training helps at 1.5B?
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
