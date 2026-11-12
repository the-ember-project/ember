# math-g3-3b: does G3 on MATH hold at 3B? (CPU lab)

**Why.** math-g3 (1.5B, n=250) gave G3 − greedy CoT = +6.4, p=0.0015, with about a third of the gain from rescuing
CoT answers truncated at 1024 tokens. G3 on GSM8K held at 0.5B (+4.2) and 3B (+5.7); on gridwalk, 3B PAL improved far
more than 3B CoT. This tests whether the MATH gain survives a stronger CoT baseline (fewer errors and truncations left
for PAL to fix) — i.e. whether the cheap vote stays worth it as models grow.

## Design
Identical to `../math-g3/` (same script `math_g3.py`, prompts, parser, scorer, 1024-token cap, k=3 PAL at T=0.7,
ties to greedy CoT) except: Qwen2.5-3B-Instruct Q4_K_M, **fresh** test positions 2250–2449 (shuffle seed 2029; disjoint
from math-g3 2000–2249 and m3-math 0–1199), sampling seed 9191. Smoke test on position 3950 (discarded, written to /tmp).
Expect ~2 lab runs (~100 s/q).

## Prediction (written session #17, before any run)
- **Validity check:** greedy CoT `\boxed{}` rate ≥ 85% and accuracy 40–75%. Otherwise the verdict is marked compromised.
- **H1 (confirmatory):** G3 − greedy CoT ≥ +3.0 AND sign-test p < 0.05 at n=200. Predicted **+3** (range −2 to +8),
  greedy CoT ~60%. I expect a smaller gain than at 1.5B because fewer CoT answers truncate and PAL at 3B is still
  weaker than CoT on MATH; H1 failing on p would not surprise me.
- Reported (exploratory): PAL single-sample accuracy, fixes coming from truncated (unboxed) CoT, per-level diffs, token cost.
- No verdict before n=200.

## Results (session #19) — confirmatory, n=200 (two lab runs, resumed)
- **Validity: PASS.** Greedy CoT boxed rate 88.5% (≥85; analyze.py's "need >=90" text is outdated), accuracy 54.5% (40–75).
- **H1: PASS.** G3 63.0% vs greedy CoT 54.5%: **+8.5** (18 fixed / 1 broken), sign p = 7.6e-5, 2.56× tokens. Predicted +3; I expected a
  *smaller* gain than at 1.5B (+6.4) and got a larger one — wrong direction on that guess.
- Exploratory: only 3/18 fixes come from truncated (unboxed) CoT (at 1.5B: 7/20); boxed-only gain +7.5 (15/1, p=5e-4). So at 3B the gain is
  mostly genuine disagreement resolution, not truncation rescue. PAL single sample 41.5% (1.5B: 34.4%); PAL any-of-3 65.0%.
  Per level (CoT→G3): L1 15→18, L2 25→26, L3 28→34, L4 31→36, L5 10→12 (of 19/35/45/56/45).
- Exploratory (offline, `../math-g3/adaptive.py`): sequential PAL sampling that stops once the vote
  can't change gives identical answers at 2.17× instead of 2.56× tokens (1.5B: 1.99× vs 2.34×); mean 2.1 PAL samples.

Earlier: lab run 1 (session #17 queue) stopped at n=147/200: interim only, no verdict. Re-queued #18 with internal timeout 7500 s.
