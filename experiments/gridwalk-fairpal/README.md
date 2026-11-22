# gridwalk-fairpal: does the gridwalk PAL advantage survive without a copyable simulator demo?

**Why.** Review 2026-11-15: in gridwalk2 the one-shot PAL demo *is* a reusable robot simulator; 52.5% of PAL samples
copy it verbatim and only swap the action list (`../gridwalk2/demo_copy.py`). So "the program format is better" may
really be "the demo hands the model a solver". Test: same 200 tasks (10 steps, task seed 778), same system prompt,
model (Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp), sampling (k=5, T=0.7, sample seed 92) and parser as gridwalk2's
`pal_s.jsonl`, but the one-shot demo is a **shop-inventory program** (event list + loop + print tuple): it shows the
format, not the direction logic. Comparisons are paired against stored gridwalk2 greedy CoT (38.0%) and original-demo
PAL vote5P (73.0%).

## Prediction (written session #24, before any run)
- **Validity:** neutral-demo PAL samples that execute to a pair ≥ 80%. If not, verdicts are marked compromised.
- **H1 (confirmatory):** vote5P(neutral) − greedy CoT ≥ +10 points AND sign-test p < 0.05 (n=200). Predicted vote5P
  ≈ 52% → **+14, pass** (~60%).
- **H2 (confirmatory):** vote5P(original demo) − vote5P(neutral) ≥ +5 AND p < 0.05. Predicted **+21, pass** (~70%).
- Exploratory: per-sample accuracy (predicted ~40% vs 57% with the original demo), share of samples that build a
  modulo-4 direction simulator on their own, tokens.
- Reading: H1 pass + H2 pass = programs help on their own, the demo adds a lot; H1 fail = the gridwalk "format" gain
  was mostly the supplied solver.

## Run
CPU lab: `run_lab.sh 240` (resumable chunks of 25; ~1000 PAL samples). Smoke-tested in session #24 on 1 task.

## Results (session #25, lab run 205 min, n=200, 1000 neutral-demo PAL samples) — confirmatory, validity FAILED
- **Validity FAIL:** only 45.2% of neutral-demo PAL samples execute to a pair (threshold ≥80%). Errors: ValueError 187,
  TypeError 122, SyntaxError 107, KeyError 33, NameError 29; ~29 run cleanly but print no pair.
- Accuracy: greedy CoT 38.0, vote5P(neutral) **2.5**, single neutral PAL 1.0 (2.0 per sample); original-demo
  vote5P 73.0, single 61.5.
- **H1 FAIL** (predicted +14): vote5P(neutral) − CoT −35.5 (4 fixed / 75 broken), p=5e-18.
- **H2 PASS** (predicted +21): vote5P(orig) − vote5P(neutral) +70.5 (143/2), p=5e-40.
- The validity failure does not rescue H1: among the 452 samples that *do* execute to a pair only 20 (4.4%) are
  right. Reading 2 samples: the model writes a plausible-looking direction helper, then does not transcribe the
  instruction list (e.g. loops `for _ in range(6)` alternating north/west, overwrites x,y each step, or sums all
  steps and walks once). 22.3% of samples build a direction-list/modulo-4 structure, but still mistranscribe.
- **Conclusion:** at 1.5B the whole gridwalk PAL advantage (+35 over CoT) comes from the demo handing over a
  solver; without it, writing code is far *worse* than CoT (2.5 vs 38). gridwalk2/m3-gridwalk results should be
  read as "the model can fill an action list into a supplied simulator", not "programs help it reason". Caveat:
  one neutral demo only; a demo that shows *list transcription* of a different domain might land in between.


## Wording correction (review 2026-11-22, session #38)
This run failed its own validity check (45% of programs executed vs ≥80% required), and only one neutral demo was
tried. What it shows: *with this neutral demo* PAL does far worse than CoT (2.5 vs 38.0). It does not show that the
gridwalk PAL gain came "entirely" from the demo's solver — a demo that teaches the program format without the simulator
might still help. The gridwalk result stays out of M2 as unproven, not as disproven. Settling it needs a second neutral
demo that passes validity (low priority).
