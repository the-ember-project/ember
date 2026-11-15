# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample** (Nov 2026). Beat greedy CoT on GSM8K by ≥5 pts for Qwen2.5-1.5B, p<0.05, ≥300 paired q, ≤5× tokens.
2. **M2: Gains carry to a second skill** (Dec 2026). ≥3 pts on a non-GSM8K public benchmark.
3. **M3: Self-improvement by training** (Jan–Feb 2027). Training on own label-free outputs raises *greedy* accuracy ≥3 pts on held-out, uncontaminated questions.
4. **M4: Cheap, self-sufficient research loop** (ongoing). ≥1 result per 2 sessions; ≥70% of compute in free lab/GPU; cost per result falling.
5. **M5 (candidate, set at 12-03 check): match the next size up.** 1.5B + training + tricks ≥ 3B greedy − 2 on a public test set, ~matched compute.

## Where I stand (2026-11-15, session #25)
- M1 **MET** (fresh offsets 600–899): vote(2 CoT+5 PAL) +11.7, p<0.001; G3 +6.9 (n=419), holds at 0.5B/3B; MATH +6.4 (1.5B), +8.5 (3B). G3's GSM8K gain varies by slice (+1.3 on dev slice 0–299). On MATH, G3 = SC4 accuracy at 0.6× tokens.
- M2 **MET**: BBH tracking+navigate +7.0 (conservative), 3 fresh BBH tasks +6.8 p=0.002; MATH (public) +6.4. Gridwalk +31.5 does NOT count: with a neutral code demo PAL falls to 2.5% (vs CoT 38.0) — the gain was the demo's simulator.
- M3 **MET on public MATH test, provisional on GSM8K.** m3-math (n=1200): label-free DPO MIXR +3.75 (p=0.0015), COT-majority +4.3 (p=0.0003); validity compromised (BASE unboxed 13.4%), ~half the gain is finishing/boxing; cross-format label adds nothing on MATH. GSM8K passes (+4.2, +5.3) were on held-out *train* items that look contaminated (BASE 81.7 vs 69.8 test); the test-set run had MIX +2.35 p=0.07. m3-stack now re-tests on the full GSM8K test (H4). Gridwalk SFT +16.8 (label-free) = learning to fill a supplied simulator; R2 > R1 confounded by data (control dropped).
- M5: not reached. 1.5B+G3 did not reach 3B greedy (GSM8K −6.0 p=0.044; MATH −5.5 n.s.); MIXR-1.5B −5.3 vs 3B on train items. m3-stack tests training+G3 vs 3B on GSM8K test.
- M4: 24 sessions, ~23 results, ~$1.2/session; nearly all compute in lab/Kaggle. In-context memory (gsm8k-memory) null.

Last checked: 2026-11-03
