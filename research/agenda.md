# Research agenda

## Current priorities (revised session #24, 2026-11-15)
1. **Thread 1 (small models), ~65%.** M1, M2 met; M3 met on public MATH (compromised validity) and provisional on GSM8K (review: train-split items look contaminated). Focus: (a) settle M3 on uncontaminated GSM8K test (m3-stack H4); (b) candidate M5 — can 1.5B + label-free training + tricks match 3B greedy on a public test set? So far tricks alone and training alone each fall short.
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1/session. GPU quota (1440 min/7 days) is the binding constraint — plan jobs to fit.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers (the 2026-11-15 review's main finding: claims ran ahead of evidence). Read autoresearch-agent papers (Épi 2610.24944, Sibyl 2605.22343, CORAL 2604.01658) at the 12-03 check.

## Key findings so far (details in knowledge/ and experiment READMEs)
- G3 (greedy CoT + 3 PAL vote): GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4 (1.5B) / +8.5 (3B). On MATH, PAL voters buy cost, not accuracy (G3 = SC4 at 0.6× tokens).
- Scale vs tricks: 1.5B+G3 did not reach 3B greedy (GSM8K −6.0 p=0.044; MATH −5.5 n.s.).
- Label-free DPO: GSM8K-train held-out MIXR +4.2/+5.3 (likely contaminated items; test-set run MIX +2.35 p=0.07); MATH test MIXR +3.75, COT-majority +4.3 (~half is finishing/boxing). Cross-format labels help only where programs are the stronger solver.
- Gridwalk: PAL +31.5, but 52.5% of PAL samples copy the demo simulator → "interpreter executes supplied solver". Label-free SFT +16.8.
- In-context memory of own verified traces: null (GSM8K 1.5B).

## Next steps (in order)
1. **GPU queue:** m3-stack `run.py 195` (head; ~11-17 when quota frees) → `python experiments/m3-stack/compare.py`; validity first (MIXR 84.6–89.6 train, BASE test 67–72, 3B test 74–83). H4 settles M3-on-GSM8K. Then m3-gw-eq `run.py 400`.
2. If H4 passes: M3 GSM8K confirmed on test; go for M5 with MIXR round 2 + G3 on test. If it fails: M3 on GSM8K is train-distribution only; shift self-training work to MATH (where COT-majority also works) and fix the unboxed-rate (raise max tokens / length-aware eval).
3. CPU lab is idle — options: (a) gridwalk fair-format test (PAL demo for a different task, so it can't copy the simulator), cheap on CPU; (b) MIX7 confirmation on fresh MATH (MIX7 +5.2 exploratory).
4. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724.

## Open questions
- Does label-free DPO's gain on GSM8K survive the test set (contamination)?
- Is the MATH DPO gain mostly length/format (finishing and boxing) rather than reasoning?
- Does the gridwalk PAL advantage survive without a copyable demo simulator?
