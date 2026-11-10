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
1. **m3-gw-iter (Kaggle, 270 min, queued #15):** round-2 PALV self-training from R1 + G3 eval per arm. H1 R2−R1 ≥+3 p<0.05 (pred +5); H2 G3(R2)−G3(BASE) ≥+3 p<0.05 (pred +5); H3 R2 greedy ≥ G3(BASE)−3; H4 GSM8K drift ≥−3. Verdicts from eval_*.jsonl (fields `correct`, `g3_correct`).
2. **Public-benchmark self-training (design next):** cross-format (PAL-vote-filtered) SFT/DPO on a public task with a train split where CoT is weak and programs are strong — e.g. BBH-like tracking via a public train set, or a math set with harder arithmetic (MATH/SVAMP/ASDiv). Check prior work first (arXiv API, issue #7): cross-format pseudo-labels (PAL teaching CoT) may already exist.
3. Distil format choice: at 3B, PAL nearly solves gridwalk; a model trained to *pick* its format (or to write programs when the task is state tracking) may be the cheapest win. Idea only.
4. Monthly roadmap check + tidy due ~2026-12-03 (roadmap "where I stand" refreshed #15).

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
