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

## Results (session #23, n=250, lab runs 2026-11-14) — confirmatory
- **Validity PASS:** CS2/CS3 boxed 88.8% (≥80); n=250 paired; G3 recomputed == stored on 249/250.
- Accuracy: greedy CoT 51.2, **G3 57.6, SC4 57.6**, MIX7 (greedy+3CS+3PAL) 62.8.
- **H1 FAIL:** G3 − SC4 = +0.0 (fixed 19 / broken 19), p=1. Predicted +1.0.
- **H2 PASS (non-inferiority):** G3 ≥ SC4 − 2 (diff 0.0).
- **H3 PASS (cost):** G3 2.34× vs SC4 4.00× greedy tokens.
- Exploratory: SC4−COT +6.4 (19/3, p=0.0009); G3−COT +6.4 (20/4, p=0.0015); MIX7−G3 +5.2 (19/6, p=0.015).
- Reading: on MATH at equal voter count, PAL voters and sampled-CoT voters give the same accuracy; PAL's advantage is cost
  (programs are ~3× shorter than CoT), not accuracy. Mixing both formats at 7 voters adds +5 more (exploratory).
  This answers the 11-08 review's equal-count concern for MATH: "mixing beats sampling" is NOT supported on MATH at
  equal count; "mixing is cheaper at equal accuracy" is.
