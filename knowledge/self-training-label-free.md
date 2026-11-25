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

## Session #16: iteration + compounding; prior work for MATH
- m3-gw-iter (gridwalk, label-free PALV SFT): R1 68.8 → R2 74.0 (+5.2, p=3e-6; R2 had 2.3× the traces, so more-data vs better-teacher not separated). G3 on R2 80.8 vs G3(base) 69.8 (+11.0); one greedy R2 pass ≥ base 4-sample vote (+4.2, p=0.07). Round-2 PALV precision fell 87→83% but coverage rose 808→1113; JOINT CoT+PAL rule rose 81→91% once CoT was trained.
- Closest prior work to cross-format pseudo-labels: **T³RL** (arXiv 2603.02203) — test-time RL where code-execution evidence upweights verified rollouts in the vote (MATH-500/AMC/AIME). **SCRL** (2603.19880), **DDRL** (2604.21327): filter weak/ambiguous consensus, add negatives. **Mixed Distillation** (2312.10730): CoT+PoT distilled from a teacher LLM. Also ScPO (2411.04109), Setlur et al. negatives (2406.14532). My distinct bits: offline SFT/DPO at 1.5B, own program samples as teacher, iteration measured.
- **Training vs parameters (m3-vs-3b, #23, confirmatory):** on m3-dpo3's 1500 held-out GSM8K-train items, MIXR-1.5B 87.1 vs 3B greedy 92.3 (−5.3, p=5e-9; BASE 81.7): label-free DPO closes 50% of the gap, like G3 does at inference. On BBH date (untrained) 3B leads by 12.4. Follow-up m3-stack: MIXR + G3 vs 3B.

## Session #24: review corrections + m3-math (public MATH)
- **Review 2026-11-15:** the GSM8K M3 passes (m3-dpo2 +4.2, m3-dpo3 +5.3) and m3-vs-3b were scored on held-out GSM8K-*train* items, which look contaminated/easy (BASE 81.7 there vs 69.8 on test; 3B 92.3 vs ~76). The only test-set run (m3-dpo) had strict MIX +2.35, p=0.07 (fail). So "M3 met on GSM8K" is **provisional** until m3-stack's added test-set eval (H4) reports. "Closes 50% of the gap" holds on train items only. m3-dpo2 also failed its BBH-date drift criterion.
- **m3-gw-iter "iterates"** → supported wording: R2 (2.3× more data, better teacher) beats R1; m3-gw-eq (queued) is the equal-data control.
- **m3-math (MATH test, n=1200, compromised validity: BASE unboxed 13.4%):** MIXR +3.75 (p=0.0015), COT-majority +4.33 (p=0.0003), MIXR−COT −0.6. Label precision MIXR 90.2 vs COT 91.8. ~Half of each gain is teaching the model to finish and box (28/2 on BASE-unboxed items). **Cross-format labels add nothing on MATH**; the "different format as teacher" rule holds only where the program format is the stronger/independent solver (GSM8K arithmetic, gridwalk).

## Update session #29 (2026-11-18): my DPO gain on MATH was format; field context
- m3-math-stack (fresh draw, n=1000, 2048-token eval): COT-majority DPO +0.8 (p=0.58), +0.1 on boxed items. The
  m3-math +4.3 was finishing/boxing under a 1024-token cap. DPO on CoT traces also cut PAL exec 58→34%.
- TTRL (2504.16084): GRPO with majority-vote reward; big MATH-500 gains on *Qwen2.5-Math-1.5B base* (32.7→73.0),
  a model that starts badly formatted. Spurious Rewards (2506.10947, ICML 2026): random rewards give Qwen2.5-Math-7B
  +21.4 on MATH-500 vs +29.1 for ground truth; gains don't transfer to Llama/OLMo — RLVR on Qwen largely elicits
  existing behaviour (e.g. code reasoning), and format. Synthesis: 2610.17824 (verifier noise in RLVR).
- Implication for my M3 work: any label-free gain on Qwen needs a **random-reward / format-only control** and an
  untruncated eval before it counts as reasoning improvement. Candidate design: GRPO on MATH-train with GOLD vs
  MAJORITY vs RANDOM reward, Qwen2.5-1.5B-Instruct, eval at 2048 tokens on fresh MATH test.

## m3-grpo-main run 3 (session #38, confirmatory) + related work flagged by review 2026-11-22
- GRPO 100 steps, LoRA, 1.5B-Instruct, MATH-train prompts, greedy 2048-tok eval on 1000 MATH-test items:
  GOLD +1.6 (n.s.), **MAJ (majority-vote reward, label-free, = TTRL-style) +3.8 p=0.004**, RANDOM +1.0. MAJ − BASE on
  BASE-boxed items +3.4 (p=0.017) → not just boxing (unlike DPO in m3-math-stack). GOLD < MAJ (−2.2, p=0.09) ⇒ likely
  ~2-pt run-to-run variance; replication `m3-grpo-rep` (2 fresh MAJ seeds, eval slice 0–999) queued.
- Related: ScPO (2411.04109) = my COT-majority DPO arm. Lin et al. 2605.19444: most TTRL gains are sharpening of problems
  the model can already solve (consistent with MAJ gaining on Levels 3–4, not 5). Gao 2606.05932: TRUE−RANDOM conflates
  self-consistency elicitation with reward design. Spurious Rewards 2506.10947 (random rewards help Qwen-Math base) —
  here RANDOM gave only +1.0 on an Instruct model.


## Status #44 (2026-11-25)
On MATH at 1.5B-Instruct, every label-free signal tried gives ≈+1–2 greedy, mostly boxing: DPO (+0.8 at 2048 tok), MAJ-GRPO (+3.8/+3.0/+1.1), RM-GRPO with Skywork-0.6B (+1.5/+2.2, = MAJ). On GSM8K, MIXR-DPO (PAL-vote labels) +3.3 on the full test (1 seed). Working hypothesis: label-free training helps when the label source is a stronger solver than the policy (PAL on GSM8K), consistent with Lin 2605.19444 (TTRL = sharpening).

## Prior work for the teacher-distillation candidate (helper search #44; verify before relying)
- Li et al. 2502.12143 (Small Models Struggle to Learn from Strong Reasoners): ≤3B students don't reliably gain from long
  CoT / large-teacher distillation; learn better from small teachers; propose Mix Distillation. No 1.5B numbers retrieved.
- U-OPSD 2608.06296: label-free on-policy self-distillation via majority-vote pseudo-solutions (Qwen3-4B; 86.7% pseudo-label precision on a 64-prompt probe).
- 2609.35259 (on- vs off-policy distillation dynamics): has Qwen2.5-1.5B-Instruct student / 7B-Instruct teacher runs — read for numbers.
- AdvDistill 2507.00054: 7B→1.5B, claims 2× over base; numbers unverified.
- GPU window (#44): ~220 min free by 11-27 17:00, ~515 by 11-28 20:00.
