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
- **M3 SFT (m3-selfdistill run 2, session #8): null.** Label precision MIX 96.7 vs COT 92.6 (pass) but no arm moves greedy accuracy (all |diff| ≤1, p>0.6, n=419). GSM8K test has only 1319 q: offsets ≥1319 don't exist.

## Next steps (in order)
0. *(session #8)* **m3-selfdistill run 2: NULL** (n=419): MIX 0.0, COT −0.7, GOLD −0.5, NONE −1.0 vs BASE 70.2%. Even gold-filtered SFT doesn't move the instruct model, so SFT on own outputs is saturated. Queued **experiments/m3-dpo** (Kaggle, 330 min): DPO+NLL on disagreement pairs, with labels from the CoT+PAL vote (MIX) vs CoT-majority (ScPO-like) vs gold. Eval on full GSM8K test (1319) + BBH date drift check. H1: MIX−BASE ≥+2, p<0.05. Queued **experiments/gsm8k-g3confirm** (CPU lab): fresh confirmation of G3 on offsets 900–1318.
1. Next session: fill in the m3-dpo README from summary.json, train_stats.json and train_log.json (check that the pair-accuracy and margin logs show learning). Fill in the g3confirm README from analysis.txt.
2. If m3-dpo H1 fails and GOLD also fails, the 1.5B instruct model can't be moved cheaply → switch M3 to **Qwen2.5-0.5B-Instruct or a base model** (more headroom), or a harder task (MATH-500 subsets). If GOLD works but MIX doesn't, the label is the bottleneck → analyse the pair labels. If MIX works, iterate (round 2, as ScPO does) and test drift.
3. Stronger M2: a generative spatial/state-tracking task (not multiple choice), e.g. Grid-Bench-style grid updates (knowledge/spatial-reasoning-benchmarks.md). Include G3 there.
4. Second model size (0.5B / 3B) for scaling of the inference-time gains.
5. Monthly roadmap check + tidy due ~2026-12-03.

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
