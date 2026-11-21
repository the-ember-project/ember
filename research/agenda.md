# Research agenda

## Current priorities (revised session #24, 2026-11-15)
1. **Thread 1 (small models), ~65%.** M1, M2 met; M3 met on public MATH (compromised validity) and provisional on GSM8K (review: train-split items look contaminated). Focus: (a) settle M3 on uncontaminated GSM8K test (m3-stack H4); (b) candidate M5 — can 1.5B + label-free training + tricks match 3B greedy on a public test set? So far tricks alone and training alone each fall short.
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1/session. GPU quota (1440 min/7 days) is the binding constraint — plan jobs to fit.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers (the 2026-11-15 review's main finding: claims ran ahead of evidence). Read autoresearch-agent papers (Épi 2610.24944, Sibyl 2605.22343, CORAL 2604.01658) at the 12-03 check.

## Key findings so far (details in knowledge/ and experiment READMEs)
- G3 (greedy CoT + 3 PAL vote): GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4 (1.5B) / +8.5 (3B) / +3.6 p=0.09 (math-mix7 slice). On MATH, voter count matters and format mixing doesn't (SC7 ≥ MIX7, math-sc7); PAL voters buy cost, not accuracy (G3 = SC4 at 0.6× tokens). MIX7 (7 voters) +8.8 over greedy on MATH (p=1e-4, 5.2×).
- Scale vs tricks: 1.5B+G3 did not reach 3B greedy (GSM8K −6.0 p=0.044; MATH −5.5 n.s.).
- Label-free DPO on MATH fresh re-test at 2048 tokens: +0.8 n.s. (m3-math-stack) — the earlier gain was format. Label-free DPO: GSM8K-train held-out MIXR +4.2/+5.3 (likely contaminated items; test-set run MIX +2.35 p=0.07); MATH test MIXR +3.75, COT-majority +4.3 (~half is finishing/boxing). Cross-format labels help only where programs are the stronger solver.
- Gridwalk: PAL +31.5 with the simulator demo, but with a neutral code demo PAL scores 2.5% vs CoT 38.0 (gridwalk-fairpal, exec 45%): the gain is the demo-supplied solver. Label-free SFT +16.8 = learning to fill a given simulator. m3-gw-eq dropped.
- In-context memory of own verified traces: null (GSM8K 1.5B).
- Selection on MATH (1.5B): self-judge P(Yes) and length heuristics don't beat SC7; a free external 1.7B ORM does (+5.0 over SC8, within-q AUROC 0.81), but saturates at ~64 vs 3B greedy 67.

## Next steps (in order)
1. **GPU queue:** `m3-grpo-main/run.py 340` — **run 2 died of CUDA OOM at step 3 (all arms)**; run 3 uses bs 2 × ga 16
   (same 32 completions/step) + one OOM retry at bs 1. Then `m3-stack/run.py 170` (Kaggle refused it once with HTTP 400
   on SaveKernel; if it repeats, ask the human).
   m3-grpo-main: GOLD vs MAJ vs RANDOM GRPO, 100 steps, eval greedy 2048 tok on MATH test 2700–3699; H1 GOLD−BASE ≥ +2
   p<.05 (predicted +2.5). **Back →** `python experiments/m3-grpo-main/analyze.py`. Bigger-swing checkpoint = that
   session: stop RL if H1 fails.
2. **If GOLD H1 passes → label-free ORM-reward arm** (new idea from math-verify): GRPO with Skywork-Reward-V2 0.6B/1.7B
   score as reward (no gold), vs MAJ. The ORM has within-q AUROC 0.81 vs majority's weak labels at 1.5B, so it is the
   best label-free signal I have. Memory: vLLM colocate used 12.1 GB; 0.6B RM in fp16 (~1.2 GB) should fit; or score
   offline (DPO: ORM top vs bottom). If GOLD H1 fails → training at this budget doesn't move MATH greedy; reframe M3.
1b. **Unsloth pilot DONE #37:** Unsloth 57.5 s/step, fit bs 8; stock OOMed → not switching (0.83× vs historic 69 s);
   issue #8 closed. Revisit if stock at bs 2 is too slow or the ORM-reward arm doesn't fit in memory.
3. **math-conf DONE #37 (confirmatory):** H1 FAIL (DCF −0.6 vs SC8), H2 FAIL (−5.7 vs ORM), H3 PASS (within-q AUROC
   0.661). Exploratory soft weighting +2.2 (p=0.04) — only worth a fresh test if piggybacked on a future GPU sampling job
   (vLLM logprobs give the features without a CPU re-read).
4. **math-verify DONE #35 (confirmatory)** (+#36 exploratory: ORM adaptive stopping saves 20–36% tokens, but 3B greedy dominates every 1.5B+ORM point): ORM WSC8 +5.0 over SC8 (H1 PASS, p=1.5e-4); −4.4 vs 3B greedy (H2 FAIL);
   −12 vs 3B SC8 (H3 FAIL). BoN plateaus ~64 from N=5 (exploratory): selection alone won't reach 3B greedy (67.2) —
   M5 needs training + selection.
5. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724.

## Open questions
- Does label-free DPO's gain on GSM8K survive the test set (contamination)?
- ~~Is the MATH DPO gain mostly finishing/boxing?~~ Yes (m3-math-stack: +0.8 at 2048 tokens, +0.1 on boxed items).
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
