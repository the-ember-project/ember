# gridwalk-analogpal: does a structurally analogous (non-grid) code demo let 1.5B write its own gridwalk simulator?

**Why.** Review 2026-11-22: gridwalk-fairpal's neutral (shop-inventory) demo failed validity (45% exec) and only one
neutral demo was tried, so "the gridwalk PAL gain is the demo's solver" is unproven. This tests the in-between demo the
fairpal README proposed: a **4-colour spinner** program — cyclic state with `% 4`, clockwise/counter-clockwise/halfway
turns, an action that depends on the current state, instructions transcribed into a list of tuples. It has the
*structure* of a direction simulator but no grid, coordinates or direction vectors: the model must map "turn left/right/
around" and "walk forward n" onto x/y itself. Everything else = gridwalk-fairpal/gridwalk2 (200 tasks, 10 steps, task
seed 778, Qwen2.5-1.5B-Instruct Q4_K_M, k=5, T=0.7, sample seed 92, same system prompt and parser). Paired against stored
gridwalk2 greedy CoT (38.0%) and original-simulator-demo PAL vote5P (73.0%); fairpal neutral vote5P was 2.5%.

## Prediction (written session #39, before any run)
- **Validity:** analog-demo PAL samples that execute to a pair ≥ 80%. If not, verdicts are marked compromised.
- **H1 (confirmatory):** vote5P(analog) − greedy CoT ≥ +10 AND sign-test p < 0.05 (n=200). Predicted vote5P ≈ 45%
  → **+7, FAIL** (~55% confident it fails; the model may copy the structure but mis-map left/right or vectors).
- **H2 (confirmatory):** vote5P(original demo) − vote5P(analog) ≥ +5 AND p < 0.05. Predicted ≈ +28, **PASS**.
- Exploratory: per-sample accuracy, exec rate, share of samples with `% 4`, comparison with fairpal's neutral 2.5%.
- Reading: H1 pass = a program *structure* from another domain transfers (the gridwalk gain is not just a handed-over
  solver, so gridwalk could count again for M2 as "format with an analogous demo"); H1 fail with valid exec = at 1.5B
  PAL on gridwalk needs the domain solver in the demo.

## Run
CPU lab: `run_lab.sh 240` (resumable chunks of 25; ~1000 PAL samples, fairpal took 205 min).

## Results (lab run 2026-11-22, analysed session #40) — confirmatory, VALIDITY FAILED
- n=200. Greedy CoT 38.0; vote5P(analog demo) **4.5**; pal1(analog) 3.0; vote5P(original solver demo) 73.0.
- Validity: only 51.0% of analog samples executed to a pair (<80%) → verdicts marked compromised.
- H1 FAIL: vote5P(analog) − CoT = −33.5 (3 fixed / 70 broken, p=1e-17). Predicted +7 FAIL; the miss is far larger than predicted.
- H2 PASS: original − analog = +68.5 (139/2, p=7e-39).
- Exploratory: among samples that *did* execute, only 3.9% are correct (20/510); fairpal's neutral demo gave 4.4%
  (20/452). So the low exec rate does not explain the failure — executing programs are wrong too. 26.6% of samples
  build a direction-list/%4 structure, but read samples ignore the instruction list (e.g. zip fixed steps with a
  fixed N/E/S/W list, or loop a constant turn).
- Reading: two different non-solver demos (neutral; structurally analogous spinner) both put 1.5B PAL ~34 pts *below*
  CoT, with ~4% accuracy even on executed programs. Both runs fail the exec validity check, so the claim stays
  "not shown", but the evidence now points one way: at 1.5B the gridwalk PAL gain depends on the demo's domain
  solver. Gridwalk stays out of M2. Answers review 2026-11-22 point 2.
