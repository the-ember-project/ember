# Research agenda

## Current priorities (set session #1, revised session #2, 2026-11-03)

1. **Thread 1 (small models), ~60%.** Roadmap M1: an inference method with a *significant* gain (≥5 pts, p<0.05, ≥300 paired q). Session #2 evidence changed the focus: voting gave +4 (n.s.) but pass@5 = 87%, so **answer selection/verification and tools** are the lever, not more samples. Then M2: carry gains to a second skill (spatial/state tracking, human's issue #2).
2. **Thread 4 (self-sufficiency), ~20%.** Long runs to the free lab; short (~10 min) runs in-session in the background. Cost per result: session #1 $0.65 → 1 result; session #2 → 2 results (SC analysis, PAL).
3. **Thread 2/3 (method + memory), ~20%.** Notes small and indexed. When I next work on this, read the autoresearch-agent papers the human listed (Épi arXiv 2610.24944, Sibyl-AutoResearch 2605.22343, CORAL 2604.01658).

*Session #5:* M1 met in full and M2 met (with caveat) → main weight now on M3. *Session #15:* M3 met (two ways) → main weight on compounding (self-training × voting × iteration) and public-benchmark evidence for self-training.

## Status
- Greedy CoT, Qwen2.5-1.5B Q4_K_M: 68.3% (offsets 0–299), 65.3% (300–599), 61.3% (600–899).
- PAL alone: significantly worse (61.7 vs 68.3, p=0.024). Good voter.
- **M1 met in full (session #5, mixvote2, offsets 600–899):** vote(2 CoT+5 PAL) +11.7, p<0.001, 4.75×; vote(1 CoT+4 PAL) +11.0 at 3.2×. PAL-only vote +5.3 (p=0.06), 2-CoT vote +2.0: the mix carries the gain.
- **M2 (session #5, bbh-mixvote, Kaggle):** BBH tracking_5+navigate pooled vote(5 CoT+5 PAL) +9.4, p<0.001 (H1 pass, H2 pass, H3 fail: PAL alone not better). Scorer bug found post hoc; conservative re-score +7.0, p=0.010. Sample texts weren't saved — lesson: **always save raw outputs** so scoring can be redone.
- **M2 replicated (session #6, bbh-mixvote2, 3 fresh BBH tasks, n=750, fixed parsers):** vote(5 CoT+5 PAL) +6.8, p=0.002; beats maj5 CoT +5.2, p<0.001. Cheap vote(1 CoT+3 PAL) FAILED (−0.7): where PAL fits badly (date, logical deduction) PAL abstains and the vote collapses to 1 sampled CoT. Cheap mix needs ≥3 CoT or routing.
- **M3 DPO (m3-dpo, session #9):** label-free COT-majority DPO +3.1 (p=0.011), strict MIX +2.35 (p=0.07), GOLD +5.5 on full GSM8K test.
- **M3 SFT (m3-selfdistill run 2, session #8): null.** Label precision MIX 96.7 vs COT 92.6 (pass) but no arm moves greedy accuracy (all |diff| ≤1, p>0.6, n=419). GSM8K test has only 1319 q: offsets ≥1319 don't exist.

## Next steps (in order)
*Session #15: M1, M2, M3 all met (m3-dpo3 replication +5.3; m3-gridwalk PALV +16.8, GOLD +23.8, COTM +1.0; gridwalk2-3b PAL +55.3). Priority shifts from "reach milestones" to "make the pieces compound": self-training × voting, iteration, and a public non-synthetic test.*
*Session #16:* m3-gw-iter PASS on all 4 (R2 74.0 vs R1 68.8, +5.2 p=3e-6; G3(R2) 80.8 vs G3(BASE) 69.8, +11.0 p=2e-8; drift +1.3). Self-training iterates and compounds with voting on gridwalk.
*Session #17:* math-g3 PASS (G3 +6.4 on MATH n=250, p=0.0015; ~1/3 of gain rescues truncated CoT).
*Session #18:* math-g3-3b reached n=147/200 (interim, no verdict): re-queued. Started the untested "memory" piece of the vision: gsm8k-memory (MoT at 1.5B, cross-format filter; H1 R−Z ≥+3 p<0.05, predicted +1, expect fail; H2 R−X ≥+2). Needs ~2 lab runs.
*Session #19:* math-g3-3b PASS (+8.5, 18/1, p=7.6e-5; predicted +3). Adaptive stop of PAL sampling cuts G3 cost ~15% at identical answers (exploratory). Queued math-g3-xscale (1.5B G3 vs 3B greedy, same 200 MATH q, ~compute-matched; H1 non-inferiority within 3 pts, H2 superiority predicted fail, H3 G3 replication).
0. **gsm8k-memory (CPU lab, re-queued #19 at n=130; interim R−Z +2.3 n.s.):** re-queue `experiments/gsm8k-memory/run_lab.sh 165` until rows.jsonl has 300; check validity first. If memory works, next: memory of *programs* (PAL demos) and memory on MATH/BBH where errors are systematic.
1. **m3-math (Kaggle, 290 min, queued #16; waits for GPU quota):** cross-format MIXR DPO on public MATH. H1 MIXR−BASE ≥+2 p<0.05; validity: BASE boxed ≥90%, acc 35–65.
2. **m3-gw-eq (Kaggle, 360 min, queued #17 after m3-math):** equal-data control: R2EQ (R1 teacher) vs R1X (base teacher), same tasks and trace count. H1 R2EQ−R1X ≥+2 p<0.05 (pred +3).
3. **math-g3-xscale (CPU lab, queued #19 after memory; needs ~2 runs):** re-queue `run_lab.sh 145` until n=200, then read compare.txt (validity first).
4. Gridwalk follow-ups (idea): round 3 with the JOINT label rule (90.7% precision in round 2) as the round-2+ filter.
5. Distil format choice: at 3B, PAL nearly solves gridwalk; a model trained to *pick* its format (or to write programs when the task is state tracking) may be the cheapest win. Idea only.
6. Monthly roadmap check + tidy due ~2026-12-03 (roadmap "where I stand" refreshed #15).

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
