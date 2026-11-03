# Research agenda

## Current priorities (set session #1, revised session #2, 2026-11-03)

1. **Thread 1 (small models), ~60%.** Roadmap M1: an inference method with a *significant* gain (≥5 pts, p<0.05, ≥300 paired q). Session #2 evidence changed the focus: voting gave +4 (n.s.) but pass@5 = 87%, so **answer selection/verification and tools** are the lever, not more samples. Then M2: carry gains to a second skill (spatial/state tracking, human's issue #2).
2. **Thread 4 (self-sufficiency), ~20%.** Long runs to the free lab; short (~10 min) runs in-session in the background. Cost per result: session #1 $0.65 → 1 result; session #2 → 2 results (SC analysis, PAL).
3. **Thread 2/3 (method + memory), ~20%.** Notes small and indexed. When I next work on this, read the autoresearch-agent papers the human listed (Épi arXiv 2610.24944, Sibyl-AutoResearch 2605.22343, CORAL 2604.01658).

*Would change this:* if PAL and selection both give < 2 pts, shift to training (LoRA self-distillation on verified answers, M3) and ask for the GPU lab.

## Status
- Greedy CoT, Qwen2.5-1.5B Q4_K_M: 68% (100 q), 68.5% (next 200), 68.3% pooled.
- Self-consistency (confirmatory): maj@5 72% vs 68%, 8/4 discordant, p=0.39 — threshold met, not significant. pass@5 87%. `experiments/gsm8k-selfconsistency/`.
- PAL (program-aided): offset 0 done in session #2: 64% vs CoT 68% (11 fixed/15 broken, n.s.); agree 59/100 with 89.8% accuracy when agreeing → agreement is a strong selection signal; offset 100–299 queued in the lab (`experiments/gsm8k-pal/run.sh`). Pre-registered: PAL ≥ CoT +3 on 300 paired q. Analyse with `python experiments/gsm8k-pal/analyze.py`.

## Next steps (in order)
1. Read PAL lab results, run `analyze.py`, fill README (confirmatory for the 300-q criterion).
2. **Selection experiment** (uses existing data, cheap): choose among CoT samples + PAL answer by (a) agreement of CoT and PAL, (b) majority over 5 CoT + PAL votes with PAL weighted, (c) "if PAL program runs and agrees with any CoT sample, take it". Pre-register on offsets 0–99, confirm on fresh offset ≥300 in the lab.
3. M2: spatial/state-tracking generator (grid navigation, swaps, gear chains) with step-count difficulty; CoT vs code simulation. Do 2–3 searches for existing spatial benchmarks first.
4. Second model size (Qwen2.5-0.5B / 3B) for scaling of gains.
5. Training (M3): LoRA self-distillation of verified answers into the 0.5B model on CPU; GPU issue if needed.

## Open questions
- Are CoT and PAL errors complementary? (Early: 10 q, agree 6/10, all agreed answers correct.)
- How much of small-model GSM8K error is arithmetic vs reasoning? PAL fixed/broken counts answer this partly.
