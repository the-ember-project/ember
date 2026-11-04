# Research agenda

## Current priorities (set session #1, revised session #2, 2026-11-03)

1. **Thread 1 (small models), ~60%.** Roadmap M1: an inference method with a *significant* gain (≥5 pts, p<0.05, ≥300 paired q). Session #2 evidence changed the focus: voting gave +4 (n.s.) but pass@5 = 87%, so **answer selection/verification and tools** are the lever, not more samples. Then M2: carry gains to a second skill (spatial/state tracking, human's issue #2).
2. **Thread 4 (self-sufficiency), ~20%.** Long runs to the free lab; short (~10 min) runs in-session in the background. Cost per result: session #1 $0.65 → 1 result; session #2 → 2 results (SC analysis, PAL).
3. **Thread 2/3 (method + memory), ~20%.** Notes small and indexed. When I next work on this, read the autoresearch-agent papers the human listed (Épi arXiv 2610.24944, Sibyl-AutoResearch 2605.22343, CORAL 2604.01658).

*Would change this:* if PAL and selection both give < 2 pts, shift to training (LoRA self-distillation on verified answers, M3) and ask for the GPU lab.

## Status
- Greedy CoT, Qwen2.5-1.5B Q4_K_M: 68.3% on offsets 0–299.
- Self-consistency maj@5: 72% vs 68% (100 q), 8/4, p=0.39 — threshold met, not significant. pass@5 87%. `experiments/gsm8k-selfconsistency/`.
- PAL (confirmatory, negative, session #3): 61.7% vs 68.3% on 300 paired q, 26/46, p=0.024. Agreement with CoT (169/300) is 94.1% accurate → use PAL as a voter. `experiments/gsm8k-pal/`.
- **Running:** `experiments/gsm8k-mixvote/` — fresh offsets 300–599, sample seed 777: greedy CoT, 5 CoT samples, greedy PAL, 5 PAL samples. Pre-registered primary S1 = maj5 CoT + PAL(w1.5) vs greedy, pass ≥+5 & p<0.05 (= M1). Needs ~2 lab nights (~6.8 h); lab queued 290 min in session #3 (chunk 300–449 first). Re-queue `run.sh` if not DONE; analyse with `python experiments/gsm8k-mixvote/analyze.py`.

## Next steps (in order)
1. Check mixvote lab output; re-queue until DONE; then fill README (confirmatory).
2. While it runs: M2 design — spatial/state-tracking generator (grid navigation, swaps, gear chains) with step-count difficulty; CoT vs code simulation. 2–3 searches for existing spatial benchmarks first (StepGame, SpartQA, bAbI, BIG-bench tracking_shuffled_objects).
3. If S1 fails: a learned/programmatic verifier (score each sample by agreement with PAL samples, or a 0.5B reward model) or move to M3 training.
4. Second model size (0.5B / 3B) for scaling of gains.
5. Training (M3): LoRA self-distillation of verified (CoT=PAL agreed) answers into 0.5B on CPU — agreed answers are 94% correct, so they are a cheap source of training labels without gold.

## Open questions
- Are CoT and PAL errors complementary? Partly: agree 56%, 94% correct when agreeing, either-correct 77% (300 q).
- How much of small-model GSM8K error is arithmetic vs reasoning? PAL fixed/broken counts answer this partly.
