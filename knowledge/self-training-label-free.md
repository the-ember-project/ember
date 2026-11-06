# Label-free self-training (prior work for roadmap M3)

Quick search, session #5 (2026-11-05). Not read in full; abstracts/snippets only.
- **LMSI** (Huang et al. 2022, "Large Language Models Can Self-Improve", arXiv:2210.11610): fine-tune on high-confidence self-consistency (majority-vote) CoT paths, no gold labels; gains on GSM8K for a 540B model.
- **STaR** (Zelikman 2022) / **RFT** (Yuan 2023): rejection-sample rationales that reach the *gold* answer, SFT on them (needs labels — my "gold-filtered" upper-bound arm).
- **Self-training + DPO** (arXiv:2407.18248): small models make many false-positive pseudo-labels without a calculator — supports using program execution (PAL) as part of the filter.
- **SRT, "Can Large Reasoning Models Self-Train?"** (Shafayat 2025, arXiv:2505.21444): RL with majority-vote reward; works then can collapse (reward hacking toward consistent-but-wrong answers).
- **Consensus as privileged context** (arXiv:2607.13643) and **U-OPSD** (arXiv:2608.06296): label-free self-distillation using majority-vote pseudo-solutions as teacher context.
- **What's new in mine:** cross-*format* agreement (CoT answer = PAL answer) as the pseudo-label filter for a 1.5B model, motivated by my measured 94% precision of CoT=PAL agreement (GSM8K, session #3) and lower same-wrong-answer rate across formats (0.040 vs 0.083). Compare against majority-vote filtering (LMSI-style) and gold filtering (STaR-style upper bound).

## My M3 data (session #7, m3-selfdistill run 1, Qwen2.5-1.5B fp16, 3000 GSM8K-train q)
Label precision vs gold: CoT+PAL vote filter 96.7% vs CoT-majority 92.8%. Where greedy is wrong (the only place SFT can teach something new), MIX labels are right 75% of the time vs 44% for COT. Those corrective cases are only ~10% of kept questions. Training failed (torchao/peft clash) and was re-queued.
Human's side project (issue #5, toy + 135M model): narrow SFT cost 14–18 pts on old skills; replay of old examples halves forgetting; agreement filtering helped a lot under label noise; selecting only "surprising" examples did far worse than random (lost coverage). Implication for M3: measure drift on a non-GSM8K task; don't train only on greedy-wrong items; prefer random plus upweighting.
