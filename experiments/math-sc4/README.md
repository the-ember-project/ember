# math-sc4: G3 vs same-format self-consistency at equal voter count (MATH, 1.5B)

**Why.** The 11-08 review said "mixing, not sampling, carries the gain" lacked an equal-count control. math-format3 H2
swapped only one voter (G3 vs CoT+2PAL+1CS: +0.4, n.s.). This is the clean control: **G3 = vote(greedy CoT + 3 PAL)**
vs **SC4 = vote(greedy CoT + 3 sampled CoT)**. Both have 4 voters and the same tie rule (ties go to greedy). The test is
on the same 250 MATH questions (math-g3, positions 2000–2249), so stored greedy CoT, PAL samples and math-format3's
first sampled CoT (CS1, seed 7373) are reused. Only 2 new sampled CoTs per question are needed (CS2, CS3).

## Design
- Qwen2.5-1.5B-Instruct Q4_K_M, llama-cpp, CoT prompt and parser from m3-math/run.py (same as math-g3/math-format3),
  T=0.7, max 1024 new tokens, **fresh seed 9191** (per-sample seed = 9191*10 + {0,5000} + j).
- SC4 = plurality of (greedy, CS1, CS2, CS3), ties to greedy, else first tied in order (same rule as math-g3 analyze).
- Cost = mean generated tokens relative to greedy CoT.

## Prediction (written session #21, before any run)
- **Validity:** CS2/CS3 boxed on ≥80% of samples; n=250 paired before any verdict.
- **H1 (confirmatory):** G3 − SC4 ≥ +2.0 AND sign-test p < 0.05. Predicted **+1.0** (range −3 to +4); expect FAIL (~30%).
- **H2 (confirmatory, non-inferiority):** G3 ≥ SC4 − 2.0. Predicted pass (~70%).
- **H3 (exploratory cost):** G3 costs less than SC4 in tokens (predicted G3 ~2.3× vs SC4 ~4.5× greedy tokens).
- Exploratory: SC4 − greedy; vote(greedy+3CS+3PAL) (7 voters); per-level.

## Results
(pending)
