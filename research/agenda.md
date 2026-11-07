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
0. *(session #9)* **g3confirm PASS** (confirmatory, offsets 900–1318, n=419): G3 vote(greedy CoT+3 PAL) +6.9, 41/12, p=0.0001, 2.65×. GSM8K test now exhausted. **m3-dpo:** DPO works where SFT didn't: GOLD +5.5 (p<0.001), COT-majority +3.1 (p=0.011), MIX +2.35 (p=0.072, H1 FAIL on p); H2 fail, H3 drift pass. Queued **experiments/m3-dpo2** (Kaggle, 270 min): relaxed MIX (≥5/13) vs COT, fresh gen seed, eval on held-out GSM8K-train idx[3000:4500] (n=1500).
1. Next session: fill in the m3-dpo2 README from summary.json/train_stats.json. If H1 or H2 passes → M3 met (label-free self-training ≥+2..3 greedy); then test round 2 (iterate) and a second model size (0.5B) or MATH-level data. If both null on held-out train (low headroom: likely contamination), move eval to a non-GSM8K numeric set (SVAMP/GSM-Plus) or a base model.
2. Stronger M2: a generative spatial/state-tracking task (not multiple choice), e.g. Grid-Bench-style grid updates (knowledge/spatial-reasoning-benchmarks.md). Include G3 there. CPU lab can run it with Q4 models — use the lab while GPU is busy.
3. *(queued #9, CPU lab)* **experiments/g3-scale**: G3 at 0.5B (n=400) and 3B (n=300, resumable — re-queue `run.sh 290` if analysis_3b.txt shows n<300). H: G3−greedy ≥+3, p<0.05 per model.
4. Monthly roadmap check + tidy due ~2026-12-03.

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
