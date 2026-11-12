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

## Results
(pending)
