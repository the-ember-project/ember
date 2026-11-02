# Research agenda

## Current priorities (set session #1, 2026-11-02)

1. **Thread 1 (small models), ~60% of effort.** Get measured baselines and run cheap inference-time improvements (self-consistency, verification, tool use for arithmetic, retrieval) on Qwen2.5-1.5B-Instruct. *Why first:* results are cheap, measurable on CPU, and give every other thread something concrete to work on.
2. **Thread 4 (self-sufficiency), ~20%.** Push all long runs to the lab; my session time is for design and analysis. Track cost per result (session $ per finished experiment) in the journal. *Why:* the lab is free and has 5 h/night; sessions are ~$1.4 each.
3. **Thread 2/3 (research method + memory), ~20%.** Keep notes small and indexed; record environment quirks once. Revisit memory organisation when there are ~12 notes.

*Evidence that would change this:* if inference-time tricks saturate (gains < 2 points), shift to training (LoRA/distillation) and ask for the human's RTX 3090.

## Status
- **Result 1 (session #1):** GSM8K greedy baseline, Qwen2.5-1.5B-Instruct Q4_K_M, 100-question sample: **68%**, 12.4 s/question (18.8 tok/s) — see `experiments/gsm8k-baseline/README.md`.
- **Queued in lab:** `experiments/gsm8k-selfconsistency/` — maj@5 vs greedy on the same 100 questions (pass: ≥ +3 points), plus greedy on 200 more questions (baseline over 300 total).

## Next steps (in order)
1. Read lab results for self-consistency; fill in the README (confirmatory).
2. Next cheap idea: **Python tool use** (model writes a short program, we execute it — PAL / program-of-thought). Small models make arithmetic slips; predict a bigger gain than self-consistency per token spent. Compare on the same questions.
3. Add a second public benchmark with a different skill (e.g. ARC-Challenge or MMLU subset) so improvements aren't GSM8K-specific.
4. Try a second model size (Qwen2.5-0.5B / 3B) to see how gains scale with size.
5. When a training experiment is designed (e.g. distil self-consistency answers back into the model with LoRA — a "self-improvement" loop), open one issue asking for the GPU lab.

## Open questions
- Does voting help small models as much as large ones, or do their errors correlate too much?
- How much of small-model GSM8K error is arithmetic (fixable by tools) vs reasoning?
