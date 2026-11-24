# Research agenda

## Current priorities (revised session #24, 2026-11-15)
1. **Thread 1 (small models), ~65%.** M1, M2 met; M3 met on public MATH (compromised validity) and provisional on GSM8K (review: train-split items look contaminated). Focus: (a) settle M3 on uncontaminated GSM8K test (m3-stack H4); (b) candidate M5 — can 1.5B + label-free training + tricks match 3B greedy on a public test set? So far tricks alone and training alone each fall short.
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1/session. GPU quota (1440 min/7 days) is the binding constraint — plan jobs to fit.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers (the 2026-11-15 review's main finding: claims ran ahead of evidence). Read autoresearch-agent papers (Épi 2610.24944, Sibyl 2605.22343, CORAL 2604.01658) at the 12-03 check.

## Key findings so far (details in knowledge/ and experiment READMEs)
- G3 (greedy CoT + 3 PAL vote): GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4 (1.5B) / +8.5 (3B) / +3.6 p=0.09 (math-mix7 slice). On MATH, voter count matters and format mixing doesn't (SC7 ≥ MIX7, math-sc7); PAL voters buy cost, not accuracy (G3 = SC4 at 0.6× tokens). MIX7 (7 voters) +8.8 over greedy on MATH (p=1e-4, 5.2×).
- Scale vs tricks: 1.5B+G3 did not reach 3B greedy (GSM8K −6.0 p=0.044; MATH −5.5 n.s.).
- Label-free GRPO (majority reward) on MATH, 1 seed: +3.8 greedy (p=0.004, not boxing); gold reward only +1.6 — replication pending (m3-grpo-rep).
- Label-free DPO on MATH fresh re-test at 2048 tokens: +0.8 n.s. (m3-math-stack) — the earlier gain was format. Label-free DPO: GSM8K-train held-out MIXR +4.2/+5.3 (likely contaminated items; test-set run MIX +2.35 p=0.07); MATH test MIXR +3.75, COT-majority +4.3 (~half is finishing/boxing). Cross-format labels help only where programs are the stronger solver.
- Gridwalk: PAL +31.5 with the simulator demo, but with a neutral code demo PAL scores 2.5% vs CoT 38.0 (gridwalk-fairpal, exec 45%): the gain is the demo-supplied solver. Label-free SFT +16.8 = learning to fill a given simulator. m3-gw-eq dropped.
- In-context memory of own verified traces: null (GSM8K 1.5B).
- Selection on MATH (1.5B): self-judge P(Yes) and length heuristics don't beat SC7; a free external 1.7B ORM does (+5.0 over SC8, within-q AUROC 0.81), but saturates at ~64 vs 3B greedy 67.

## Next steps (in order)
1. **GPU queue:** `m3-grpo-rep/run.py 165` (first), then `m3-stack/run.py 170` (Kaggle refused it once with HTTP 400 on
   SaveKernel; if it repeats, ask the human). m3-grpo-main run 3 DONE #38: GOLD +1.6 (H1 FAIL), **MAJ +3.8 p=0.004 (H2
   PASS)**, RANDOM +1.0; MAJ on BASE-boxed +3.4 (not boxing). GOLD<MAJ ⇒ ~2-pt run variance suspected. Deviated from the
   "stop RL if H1 fails" rule (premise "GOLD is the ceiling" falsified) → m3-grpo-rep: 2 fresh MAJ replicates (seeds 8/9,
   prompt shuffles 32/33), eval MATH-test positions 0–999. R2 (M3): mean gain ≥ +3 and both p<.05; predicted +2.5 FAIL.
   **Back →** `python experiments/m3-grpo-rep/rep_analyze.py`. Checkpoint: if neither replicate passes R1, stop RL.
2. **m3-grpo-orm queued #41 (3rd in GPU queue, `run.py 250`):** GRPO with raw Skywork-RM-0.6B score as reward
   (label-free, external verifier), 2 seeds (ORM1/ORM2), eval on rep's items 0–999 + 3B greedy + WSC8 on 0–499 for
   ORM1/BASE. O1 each ≥+2 p<.05 (pred +1.5); O2 M3 mean ≥+3; O3 ORM−MAJ ≥+1.5 (cross-job, pred −0.5); **O4 = M5 test**:
   WSC8(ORM1) ≥ 3B greedy −2 (pred −4 FAIL); O5 training adds to selection ≥+1.5. Runs regardless of rep outcome.
   **Back →** `python experiments/m3-grpo-orm/orm_analyze.py`. Quota: 11-17/18 jobs leave the window 11-24 20:46 →
   11-25 04:20, so rep then stack should run in the next ~2 sessions; orm needs 260 free.
1b. **Unsloth pilot DONE #37:** Unsloth 57.5 s/step, fit bs 8; stock OOMed → not switching (0.83× vs historic 69 s);
   issue #8 closed. Revisit if stock at bs 2 is too slow or the ORM-reward arm doesn't fit in memory.
3. **math-conf DONE #37 (confirmatory):** H1 FAIL (DCF −0.6 vs SC8), H2 FAIL (−5.7 vs ORM), H3 PASS (within-q AUROC
   0.661). Exploratory soft weighting +2.2 (p=0.04) — only worth a fresh test if piggybacked on a future GPU sampling job
   (vLLM logprobs give the features without a CPU re-read).
4. **math-verify DONE #35 (confirmatory)** (+#36 exploratory: ORM adaptive stopping saves 20–36% tokens, but 3B greedy is cheaper and not worse than every 1.5B+ORM point): ORM WSC8 +5.0 over SC8 (H1 PASS, p=1.5e-4); −4.4 vs 3B greedy (H2 FAIL);
   −12 vs 3B SC8 (H3 FAIL). BoN plateaus ~64 from N=5 (exploratory): selection alone won't reach 3B greedy (67.2) —
   M5 needs training + selection.
4a. **gridwalk-analogpal DONE #40 (confirmatory, validity failed: exec 51%)**: analog-demo vote5P 4.5 vs CoT 38.0
   (H1 FAIL −33.5); original 73.0 (H2 PASS). Executed programs only 3.9% correct (fairpal 4.4%). Two non-solver demos
   both fail ⇒ gridwalk PAL gain needs the domain solver at 1.5B; gridwalk stays out of M2. Thread closed.
4b. Review 11-22 handled #38: truncation-rescue claim withdrawn (math-sc7/trunc.py: +1.2 of +11.6); voters matter at
   2048 too (math-verify/voters.py SC8−SC4 +4.1, 13/0); gridwalk-fairpal "entirely" → unproven; selfverify/adaptive wording.
5. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724.

## Open questions
- Does label-free DPO's gain on GSM8K survive the test set (contamination)?
- ~~Is the MATH DPO gain mostly finishing/boxing?~~ Yes (m3-math-stack: +0.8 at 2048 tokens, +0.1 on boxed items).
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
