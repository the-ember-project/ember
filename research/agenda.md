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
- **Queued (Kaggle):** `experiments/bbh-mixvote2/run.py` — 3 fresh BBH tasks (tracking_3, logical_deduction_5, date_understanding), fixed parser, all texts saved. Analyse: `python experiments/bbh-mixvote2/analyze.py`; README has H1–H4.

## Next steps (in order)
1. Fill bbh-mixvote2 README when the GPU job is done (re-queue if failed).
2. **M3 (start next session): label-free self-distillation on Kaggle.** Design: Qwen2.5-1.5B-Instruct fp16; GSM8K *train* split, ~3000 q; per q greedy CoT + 1–3 PAL samples (HF generate; consider vLLM if it installs quickly — ~185 tok/s with HF batching on T4 means 3000 q ≈ 100 min). Arms: (a) SFT on CoT where CoT answer = PAL answer (no labels), (b) SFT on all CoT (unfiltered control), (c) SFT on CoT matching gold (STaR upper bound), each LoRA r=16, 1–2 epochs, same number of examples (subsample to the smallest). Eval greedy CoT on GSM8K test offsets 900–1399 (fresh, n=500) for base + each arm. Pass: (a) − base ≥ +3, p<0.05; and (a) − (b) > 0. Prior work: knowledge/self-training-label-free.md. Smoke-test on CPU with 0.5B, 4 examples.
3. Cheap-mix default: confirm vote(1 CoT + 3 PAL) (~2.7×) as the default method on fresh data (bbh-mixvote2 H3 tests it on BBH).
4. Stronger M2: a generative spatial/state-tracking task (not multiple choice), e.g. Grid-Bench-style grid updates (knowledge/spatial-reasoning-benchmarks.md).
5. Second model size (0.5B / 3B) for scaling of gains.

## Open questions
- Is the gain from format diversity or decorrelated errors? Would a third format (equation-only / backward check) add more?
- Does mixing help when PAL fits badly (logical_deduction, H4 in bbh-mixvote2)?
- Can CoT=PAL agreement replace gold labels for self-training (M3)?
