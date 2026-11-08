# Research agenda

## Current priorities (set session #1, revised session #2, 2026-11-03)

1. **Thread 1 (small models), ~60%.** Roadmap M1: an inference method with a *significant* gain (≥5 pts, p<0.05, ≥300 paired q). Session #2 evidence changed the focus: voting gave +4 (n.s.) but pass@5 = 87%, so **answer selection/verification and tools** are the lever, not more samples. Then M2: carry gains to a second skill (spatial/state tracking, human's issue #2).
2. **Thread 4 (self-sufficiency), ~20%.** Long runs to the free lab; short (~10 min) runs in-session in the background. Cost per result: session #1 $0.65 → 1 result; session #2 → 2 results (SC analysis, PAL).
3. **Thread 2/3 (method + memory), ~20%.** Notes small and indexed. When I next work on this, read the autoresearch-agent papers the human listed (Épi arXiv 2610.24944, Sibyl-AutoResearch 2605.22343, CORAL 2604.01658).

*Session #5:* M1 met in full and M2 met (with caveat) → main weight now on M3 (self-training on Kaggle GPU); M2 replication runs alongside.

## Status
- Greedy CoT, Qwen2.5-1.5B Q4_K_M: 68.3% (offsets 0–299), 65.3% (300–599), 61.3% (600–899).
- PAL alone: significantly worse (61.7 vs 68.3, p=0.024). Good voter.
- **M1 met in full (session #5, mixvote2, offsets 600–899):** vote(2 CoT+5 PAL) +11.7, p<0.001, 4.75×; vote(1 CoT+4 PAL) +11.0 at 3.2×. PAL-only vote +5.3 (p=0.06), 2-CoT vote +2.0: the mix carries the gain.
- **M2 (session #5, bbh-mixvote, Kaggle):** BBH tracking_5+navigate pooled vote(5 CoT+5 PAL) +9.4, p<0.001 (H1 pass, H2 pass, H3 fail: PAL alone not better). Scorer bug found post hoc; conservative re-score +7.0, p=0.010. Sample texts weren't saved — lesson: **always save raw outputs** so scoring can be redone.
- **M2 replicated (session #6, bbh-mixvote2, 3 fresh BBH tasks, n=750, fixed parsers):** vote(5 CoT+5 PAL) +6.8, p=0.002; beats maj5 CoT +5.2, p<0.001. Cheap vote(1 CoT+3 PAL) FAILED (−0.7): where PAL fits badly (date, logical deduction) PAL abstains and the vote collapses to 1 sampled CoT. Cheap mix needs ≥3 CoT or routing.
- **M3 DPO (m3-dpo, session #9):** label-free COT-majority DPO +3.1 (p=0.011), strict MIX +2.35 (p=0.07), GOLD +5.5 on full GSM8K test.
- **M3 SFT (m3-selfdistill run 2, session #8): null.** Label precision MIX 96.7 vs COT 92.6 (pass) but no arm moves greedy accuracy (all |diff| ≤1, p>0.6, n=419). GSM8K test has only 1319 q: offsets ≥1319 don't exist.

## Next steps (in order)
0. *(session #10)* **m3-dpo2: H1 PASS** MIXR +4.2 (p<0.001, held-out train n=1500) but **H2 COT −7.9** (p<0.001; m3-dpo's +3.1 didn't replicate) and **H4 drift FAIL** (MIXR −4.8, n.s.). Label-free DPO is unstable across data draws. Queued **m3-dpo3** (Kaggle 240 min): identical replication, gen seed 4041, eval idx[4500:6000]. Review fixes: BBH equal-count control shows mixing ≈ maj5 CoT at equal samples (claim corrected); m3-dpo "negatives" claim relabelled as confounded; g3-scale 0.5B run 1 invalid (prompt gave no CoT) → re-run with `--system` tutor prompt on offsets 400–799 + 3B resume to n=300 queued in CPU lab (`g3-scale/run2.sh`).
1. Next session: fill in g3-scale (analysis_0.5b_v2.txt, analysis_3b.txt; H2 judged at n=300) and m3-dpo3 if back. If m3-dpo3 H1 passes → M3 met with drift caveat; then NLL-only control on the same pairs (reviewer) and a second drift task. If it fails → M3 "not robust"; diagnose COT collapse (wrong chosen answers? margin too large → lower lr/β sweep).
2. Stronger M2: a generative spatial/state-tracking task (not multiple choice), e.g. Grid-Bench-style grid updates (knowledge/spatial-reasoning-benchmarks.md). Include G3 there. CPU lab can run it with Q4 models.
3. Monthly roadmap check + tidy due ~2026-12-03.

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
