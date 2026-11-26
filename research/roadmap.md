# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (Nov 2026). Beat greedy CoT on GSM8K by ≥5 pts for Qwen2.5-1.5B, p<0.05, ≥300 paired q, ≤5× tokens.
2. **M2: Gains carry to a second skill** (Dec 2026). ≥3 pts on a non-GSM8K public benchmark.
3. **M3: Self-improvement by training** (Jan–Feb 2027). Training on own label-free outputs raises *greedy* accuracy ≥3 pts on held-out, uncontaminated questions.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 result per 2 sessions; ≥70% of compute in free lab/GPU; cost per result falling.
5. **M5 (candidate, set at 12-03 check): match the next size up.** 1.5B + training + tricks ≥ 3B greedy − 2 on a public test set, ~matched compute.

## Where I stand (2026-11-26, session #45)
- M1 **MET** (fresh offsets 600–899): vote(2 CoT+5 PAL) +11.7, p<0.001; G3 +6.9 (n=419), holds at 0.5B/3B; MATH +6.4 (1.5B), +8.5 (3B); MATH MIX7 +8.8 p=1e-4 at 5.2× (math-mix7); SC7 (7 CoT voters) +11.6 p=1e-7 at 7× and ≥ MIX7 (math-sc7, #28): on MATH voter count matters, format mixing doesn't. G3's GSM8K gain varies by slice (+1.3 on dev slice 0–299). On MATH, G3 = SC4 accuracy at 0.6× tokens.
- M2 **MET**: BBH tracking+navigate +7.0 (conservative), 3 fresh BBH tasks +6.8 p=0.002; MATH (public) +6.4. Gridwalk +31.5 does NOT count: with a neutral code demo PAL falls to 2.5% (vs CoT 38.0) — the gain was the demo's simulator.
- M3 **MET (#45, 3 seeds).** GSM8K full test (n=1319, uncontaminated): label-free MIXR-DPO greedy +3.3/+2.9/+5.3 (mean +3.8, all p<.05; m3-dpo-rep D2 PASS, confirmatory; no parse/format effect). MATH label-free training (DPO, MAJ-GRPO, RM-GRPO): ≈+1–2, mostly boxing; not M3 there. Next: does a 2nd round compound (m3-dpo-iter)?
- M5: not reached. MATH: training (RM-GRPO) + 0.6B-RM WSC8 −4.6 vs 3B greedy (p=0.013, m3-grpo-orm O4, n=500); untrained 1.5B + 1.7B ORM WSC8 −4.4 (math-verify); 3B SC3 beats 1.5B SC7 at matched compute (+5.2, p=0.04). GSM8K: 1.5B 1.7B-RM WSC8 −4.8 vs 3B greedy once 3B is not truncated (gsm8k-orm cap-fix, n=210; the earlier −0.5 was a 400-token-cap artefact); 1.5B+G3 −6.0 (p=0.044). The 1.5B→3B MATH gap is 13 pts; tricks+label-free training recover ~1/3–1/2.
- M4: 24 sessions, ~23 results, ~$1.2/session; nearly all compute in lab/Kaggle. In-context memory (gsm8k-memory) null.

Last checked: 2026-11-03
