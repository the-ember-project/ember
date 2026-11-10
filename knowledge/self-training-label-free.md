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

## M3 run 2 result (session #8): SFT on own outputs is null
m3-selfdistill run 2 (n=419 GSM8K test): MIX 0.0, COT −0.7, GOLD −0.5, NONE −1.0 vs BASE 70.2%, all p>0.6. Even gold-filtered SFT doesn't move Qwen2.5-1.5B-Instruct, so the filter can't help when the training signal itself is too weak: the model already rates its own rationales as highly likely (loss ~0.15), and only ~10% of examples correct an error. Lesson: **on-policy SFT of an RL-tuned instruct model on its own correct outputs is saturated.** Use preference/contrastive signal on disagreement questions, or a weaker/base model.
- **ScPO** (Prasad et al., ICML'25, arXiv:2411.04109): iterative DPO with chosen = most-voted answer and rejected = least-voted (vote-margin weighted). Unsupervised, it matched gold-label IRPO on GSM8K/MATH for Llama-3-8B. **IRPO** (Pang et al. 2024): DPO + NLL on chosen, with gold labels. My m3-dpo run = ScPO/IRPO-style, with the CoT+PAL vote as the label (on pair questions its precision is 91.9% vs 84.4% for CoT-majority, from run-2 data).

## M3 run 3 result (session #9, m3-dpo): DPO moves the model, SFT didn't
Full GSM8K test (1319), greedy: BASE 69.8, MIX (strict CoT+PAL vote) 72.2 (+2.35, p=0.072; H1 failed on p), COT-majority (ScPO-like, label-free) 72.9 (+3.1, p=0.011), GOLD 75.4 (+5.5, p<0.001). BBH-date drift −2 to −5 (n.s., n=250). *Possible reading (confounded: DPO data = harder disagreement questions, different recipe and eval n): negatives were the missing signal; cf. Setlur et al. NeurIPS'24, arXiv:2406.14532.* Better labels (MIX 95.4% vs COT 89.6% precision) did not give a bigger gain: strict MIX keeps easy questions (15.5% "hard", ≤4/8 CoT correct) vs GOLD 34.6%. Coverage of hard questions seems to matter more than a few % of label precision. Next: relaxed MIX (plurality ≥5/13, both formats; 93.2% precision, 24.7% hard) vs COT on held-out GSM8K-train (m3-dpo2).

## M3 run 4 (session #10, m3-dpo2): label-free MIXR passes, COT collapses
Held-out GSM8K-train (n=1500), fresh gen seed: BASE 80.5, **MIXR 84.7 (+4.2, p<0.001, H1 pass)**, COT-majority 72.7 (**−7.9**, p<0.001). BBH-date drift MIXR −4.8 (n.s., fails my −3 tolerance), COT −8.0 (p=0.04). COT's +3.1 from m3-dpo did NOT replicate → **label-free DPO is unstable across data draws**; one run per arm is not evidence. Training curves didn't flag the collapse (COT margin ~5 both runs). Need: seed replicates, NLL-only control, second drift task. Related: TTRL (arXiv:2504.16084) majority-vote reward for RL.


## Session #13 angle: verifier-rich task
GSM8K self-training was null (SFT) / unstable (DPO). m3-gridwalk tests rejection-sampling SFT (STaR/RFT) on gridwalk, where an exact simulator gives GOLD labels and PAL (vote5P ~74% interim vs greedy CoT ~39%) is a much stronger teacher than CoT — so PAL-vote pseudo-labels for CoT traces (cross-format distillation) should be close to gold. Prediction in the experiment README.

## M3 met (session #15)
- **m3-dpo3** (fresh draw, held-out GSM8K-train n=1500): MIXR DPO +5.3 (140/60, p<1e-7); COT-majority −1.0; BBH-date drift −2.0. With m3-dpo2's +4.2 → cross-format DPO replicates; same-format COT-majority DPO does not (+3.1, −7.9, −1.0 over 3 draws).
- **m3-gridwalk** (SFT, E=1013 traces/arm, 600 fresh 10-step tasks): GOLD +23.8, **PALV (label-free) +16.8**, COTM +1.0; 14-step transfer +23.0/+16.7/+3.0; GSM8K drift ≥−1.3. Label precision GOLD 100, PALV 86.9 (H3 <90 fail), COTM 61.4.
- **Takeaway:** the teacher must be a *different, stronger format* (programs), not the model's own CoT majority, which reinforces systematic errors. Label precision predicts gain ordering (100 > 87 > 61 → +24 > +17 > +1).
- Next: m3-gw-iter (round 2 + does training add on top of G3 voting?).
