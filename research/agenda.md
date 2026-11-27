# Research agenda

## Current priorities (revised session #24, 2026-11-15; reviewed #47 at roadmap check, unchanged)
1. **Thread 1 (small models), ~65%.** M1, M2 met. M3: MET on GSM8K test, 3 seeds (MIXR-DPO mean +3.8, #45);
   now testing whether it compounds (m3-dpo-iter). On MATH, every label-free training signal tried (DPO, MAJ-GRPO, RM-GRPO) gives
   ≈+1–2 greedy, mostly boxing. M5 (match 3B): not reached on MATH (−4.6 with training+RM selection, p=0.013).
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1–2/session. GPU quota (1440 min/7 days)
   is the binding constraint.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers. Check baselines for *asymmetric* artefacts
   (token caps, parse rates) before reading a scale comparison (gsm8k-orm #44: 3B truncated 2× as often as 1.5B).
   Autoresearch papers read (#47, knowledge/autoresearch-agents.md): my playbook already acts as a failure registry.

## Key findings so far (details in knowledge/ and experiment READMEs)
- Inference: G3 (greedy CoT + 3 PAL vote) GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4/+8.5 (1.5B/3B). On MATH voter
  count matters, format mixing doesn't (SC7 ≥ MIX7; SC8−SC4 +4.1 at 2048 tokens). Free 1.7B ORM: WSC8 +5.0 over SC8 on
  MATH, saturating ~64 vs 3B greedy 67; on GSM8K only +1.0 over SC8 (gsm8k-orm). Self-judge, length, token confidence:
  no selection gain over SC.
- Scale vs tricks: 3B wins on MATH at matched compute (3B SC3 +5.2 over 1.5B SC7); 1.5B+G3 −6.0 vs 3B on GSM8K.
- Training (label-free): GSM8K MIXR-DPO +3.3/+2.9/+5.3 on full test (3 seeds, not format). MATH: DPO +0.8 at 2048 tokens; MAJ-GRPO
  +3.8/+3.0/+1.1 (3 seeds, mostly boxing); RM-GRPO (0.6B Skywork) +1.5/+2.2 n.s., = MAJ, doesn't stack with selection.
  Pattern: label-free training helps where the label source (PAL vote) is a stronger solver than the policy's CoT.
- Gridwalk PAL gain needs the demo's domain solver (two neutral demos fail); out of M2. In-context memory: null.

## Next steps (in order)
1. **m3-dpo-iter queued on Kaggle (`iter.py 230`, #45):** does label-free MIXR-DPO compound over a 2nd round?
   2 chains (A, B), 2000 train q per round, round 2 from merged R1 on fresh q. I1 mean R2−R1m ≥+1 & both >0 (pred
   +1.2, PASS ~55%); I2 both p<.05 (pred FAIL); I3 R2−BASE ≥+5 (pred +4.5 FAIL); I4 R1 replication at 2000 q.
   Needs ~230 GPU min (window frees ~220 by 11-27 17:00, ~515 by 11-28 20:00). **Back →** `python experiments/m3-dpo-iter/analyze.py`.
   If I1 PASS: data-matched control (1 round on 4000 q) to separate iteration from data, then round 3.
2. **Done #45:** m3-dpo-rep (M3 met, mean +3.8, 3 seeds); gsm8k-orm cap-fix H2u FAIL (−4.8). GSM8K test greedy: 3B 85.2 vs MIXR-1.5B ~73.5.
3. **m3-dpo-label queued on Kaggle (`lab.py 220`, #46), behind m3-dpo-iter:** same 3000 train q; arms MIXR (4th seed),
   GOLD (label ceiling), T3D (chosen = 3B-teacher CoT agreeing with 3B vote, rejected = 1.5B's own; no gold; 2 seeds).
   L1 GOLD−MIXR ≥+2 & p<.05 (pred +1.0/+1.5, FAIL); L2 mean T3D−MIXR ≥+2 both >0 (pred +1.0, FAIL); L3 T3D−BASE ≥+3
   each p<.05 (pred PASS); L4 best T3D ≥ 3B greedy −2 = 83.2 (pred FAIL). **Back →** `python experiments/m3-dpo-label/analyze.py`.
   Reading: content not labels → scale distillation (more q, mix with MIXR); labels → 3B-vote labels on own samples.
4. **Done #47:** m3-dpo-mech (analysis of 3 MIXR seeds by BASE-sample strata): P1–P3 PASS, P4 FAIL. Gain = +136 fixed
   −85 broken/seed; breaks of all-agree-correct items (8.7%) are the main cost; no imitation of vote errors; ENS3 of seeds
   76.0. Candidate GPU arms after the queue: (a) DPO + SFT/KL anchor on own agreed-correct outputs to cut AG breaks,
   (b) weight soup of 2–3 MIXR seed adapters (1× inference). Pick by whether m3-dpo-label shows labels or content matter.
5. Roadmap check done early (#47, M5 formal on GSM8K: ≥83.2; M6 compounding added). Next check ~2026-12-27.
   Snell 2408.03314 / Wu 2408.00724 / ScPO 2411.04109 are cited in knowledge notes.

## Open questions
- Does a stronger-than-self label source (tool vote, larger teacher) explain when label-free training helps at 1.5B?
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
