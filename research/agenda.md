# Research agenda

## Current priorities (set session #1, revised session #2, 2026-11-03)

1. **Thread 1 (small models), ~60%.** Roadmap M1: an inference method with a *significant* gain (≥5 pts, p<0.05, ≥300 paired q). Session #2 evidence changed the focus: voting gave +4 (n.s.) but pass@5 = 87%, so **answer selection/verification and tools** are the lever, not more samples. Then M2: carry gains to a second skill (spatial/state tracking, human's issue #2).
2. **Thread 4 (self-sufficiency), ~20%.** Long runs to the free lab; short (~10 min) runs in-session in the background. Cost per result: session #1 $0.65 → 1 result; session #2 → 2 results (SC analysis, PAL).
3. **Thread 2/3 (method + memory), ~20%.** Notes small and indexed. When I next work on this, read the autoresearch-agent papers the human listed (Épi arXiv 2610.24944, Sibyl-AutoResearch 2605.22343, CORAL 2604.01658).

*Session #4:* M1 met on effect (mixvote) → shift weight to M2 (transfer to BBH state-tracking) and M3 (training on Kaggle GPU).

## Status
- Greedy CoT, Qwen2.5-1.5B Q4_K_M: 68.3% (offsets 0–299), 65.3% (300–599).
- PAL alone: significantly worse (61.7 vs 68.3, p=0.024). Good voter.
- **Mixvote (confirmatory, session #4, offsets 300–599):** maj5 CoT +5.7 (p=0.024, 4.9×); maj5 CoT+PAL(w1.5) +8.7 (p=0.001, 5.5×); vote 5 CoT+5 PAL +13.7 (p<0.001, 7.7×). Lesson: mixing reasoning formats > more samples of one. M1 met on effect.
- **Running (lab):** `experiments/gsm8k-mixvote2/` — offsets 600–899, seed 4242: greedy CoT, 2 CoT + 5 PAL samples. Primary P1 = vote(2 CoT+5 PAL) ≥+5, p<0.05, ≤5× (full M1). Analyse with `python experiments/gsm8k-mixvote2/analyze.py`; re-queue `run.sh` if not DONE.

## Next steps (in order)
1. Fill mixvote2 README (confirmatory) when DONE.
2. **M2 = does format-mixing transfer?** QUEUED on Kaggle GPU (session #4): `experiments/bbh-mixvote/run.py` (HF fp16, T4); analyse with `python experiments/bbh-mixvote/analyze.py`, README prediction fixed (H1 pooled S2 ≥+3, p<0.05). If the job failed, fix and re-queue. Run CoT vs PAL vs mixed vote on public BIG-Bench Hard state-tracking/spatial tasks (`navigate`, `tracking_shuffled_objects_five_objects`, maybe `date_understanding`; HF `lukaemon/bbh`, 250 q each). Fits the human's issue #2 (spatial / world models) and is a public benchmark. Pre-register: mixed vote ≥ greedy CoT +3 on pooled ≥300 q.
3. Why does mixing help? Error correlation analysis: P(CoT wrong & PAL wrong same answer) vs within-format. Cheap, from existing jsonl.
4. M3: self-distil CoT=PAL agreed answers (94% correct) into 0.5B/1.5B via LoRA on Kaggle GPU (now approved, 15 h/week). Target: greedy +3 on held-out.
5. Second model size (0.5B / 3B) for scaling of gains.

## Open questions
- Is the gain from format diversity or just from decorrelated errors? Would a third format (e.g. equation-only / backward check) add more?
- Do mixed votes help when PAL is a bad fit for the task (non-numeric answers)?
