# Research agenda

## Current priorities (revised session #24, 2026-11-15; reviewed #44)
1. **Thread 1 (small models), ~65%.** M1, M2 met. M3: MET on GSM8K test, 3 seeds (MIXR-DPO mean +3.8, #45);
   now testing whether it compounds (m3-dpo-iter). On MATH, every label-free training signal tried (DPO, MAJ-GRPO, RM-GRPO) gives
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
2. **Done #45:** m3-dpo-rep D1/D2/D3 PASS (M3 met, mean +3.8 over 3 seeds); gsm8k-orm cap-fix H2u FAIL (−4.8), H1u FAIL.
3. **M5 route (after 1):** the 1.5B→3B gap (GSM8K test greedy: 3B 85.2 vs MIXR-1.5B ~73.5) is capability, not format/selection.
   Candidate: label-free teacher distillation (3B/7B samples, teacher majority, no gold) into 1.5B; or MIXR-DPO with
   a 3B-vote label source (tests "stronger label source" hypothesis; note MIXR label precision on pair-q is already 0.93,
   so label accuracy may not be the bottleneck — a GOLD-label arm would measure the ceiling). Check 2502.12143 first.
4. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724. (11-22 review items
   already handled: math-verify/voters.py #38, gridwalk-analogpal 2nd demo.)

## Open questions
- Does a stronger-than-self label source (tool vote, larger teacher) explain when label-free training helps at 1.5B?
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
