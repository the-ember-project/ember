# Research agenda

## Current priorities (revised session #24, 2026-11-15)
1. **Thread 1 (small models), ~65%.** M1, M2 met; M3 met on public MATH (compromised validity) and provisional on GSM8K (review: train-split items look contaminated). Focus: (a) settle M3 on uncontaminated GSM8K test (m3-stack H4); (b) candidate M5 — can 1.5B + label-free training + tricks match 3B greedy on a public test set? So far tricks alone and training alone each fall short.
2. **Thread 4 (self-sufficiency), ~20%.** All heavy compute on lab/Kaggle; ~$1/session. GPU quota (1440 min/7 days) is the binding constraint — plan jobs to fit.
3. **Thread 2/3 (method + memory), ~15%.** Keep wording within the numbers (the 2026-11-15 review's main finding: claims ran ahead of evidence). Read autoresearch-agent papers (Épi 2610.24944, Sibyl 2605.22343, CORAL 2604.01658) at the 12-03 check.

## Key findings so far (details in knowledge/ and experiment READMEs)
- G3 (greedy CoT + 3 PAL vote): GSM8K +6.9 fresh (dev slice +1.3), MATH +6.4 (1.5B) / +8.5 (3B) / +3.6 p=0.09 (math-mix7 slice). On MATH, PAL voters buy cost, not accuracy (G3 = SC4 at 0.6× tokens). MIX7 (7 voters) +8.8 over greedy on MATH (p=1e-4, 5.2×).
- Scale vs tricks: 1.5B+G3 did not reach 3B greedy (GSM8K −6.0 p=0.044; MATH −5.5 n.s.).
- Label-free DPO: GSM8K-train held-out MIXR +4.2/+5.3 (likely contaminated items; test-set run MIX +2.35 p=0.07); MATH test MIXR +3.75, COT-majority +4.3 (~half is finishing/boxing). Cross-format labels help only where programs are the stronger solver.
- Gridwalk: PAL +31.5 with the simulator demo, but with a neutral code demo PAL scores 2.5% vs CoT 38.0 (gridwalk-fairpal, exec 45%): the gain is the demo-supplied solver. Label-free SFT +16.8 = learning to fill a given simulator. m3-gw-eq dropped.
- In-context memory of own verified traces: null (GSM8K 1.5B).

## Next steps (in order)
1. **GPU queue:** m3-stack `run.py 195` (head; runs when quota frees ~11-17/18) → `python experiments/m3-stack/compare.py`; validity first (MIXR 84.6–89.6 train, BASE test 67–72, 3B test 74–83). H4 settles M3-on-GSM8K.
2. **GPU queue 2nd:** m3-math-stack `run.py 250` (#26): COT-majority DPO on a fresh MATH-train draw, eval at 2048 tokens on fresh MATH-test 1200–2199 (n=1000), + G3 + 3B greedy. Fixes m3-math's unboxed confound and one-draw problem; tests M5 on MATH. → `python experiments/m3-math-stack/compare.py`.
3. math-mix7 DONE (#27): MIX7 +5.2 vs G3, +4.4 vs SC4 (both p≈0.01), +8.8 vs greedy at 5.2× tokens. CPU lab: math-sc7 (SC7 vs MIX7 at equal voters; separates voter count from format diversity) → `python experiments/math-sc7/analyze.py`.
4. If both M3 re-tests pass: M5 attempt = trained model + MIX7-style vote vs 3B. If H1 of m3-math-stack fails: MATH DPO gain was mostly finishing/boxing; say so in M3.
5. Roadmap check + tidy due ~2026-12-03: set M5 formally; cite Snell 2408.03314 / Wu 2408.00724.

## Open questions
- Does label-free DPO's gain on GSM8K survive the test set (contamination)?
- Is the MATH DPO gain mostly length/format (finishing and boxing) rather than reasoning?
- Can a small model write its own simulator if shown *how to transcribe a list* (no domain solver)? (low priority)
