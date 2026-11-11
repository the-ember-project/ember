# math-g3: does the cheap G3 vote (greedy CoT + 3 PAL samples) carry from GSM8K to MATH? (CPU lab)

**Why.** G3 = plurality of [greedy CoT, 3 PAL samples at T=0.7], ties to greedy CoT, gave +6.9 on GSM8K (p=1e-4,
2.65× tokens), +4.2 at 0.5B, +5.7 at 3B, +33.5 on gridwalk. MATH (public, harder, answers often fractions; programs
can use sympy) is the natural next public test for M1/M2, and tells whether PAL is a useful teacher for m3-math.
Prior work: PAL/PoT papers report program gains on MATH for large models; XoT (2310.14628) switches CoT/PoT on MATH.

## Design
Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp, 4 threads (CPU lab). MATH test, numeric-gold subset (same loader, prompts and
parser as `../m3-math/run.py`), shuffle seed 2029, positions 2000–2249 (disjoint from m3-math's eval positions 0–1199).
max 1024 new tokens. Sampling seed 8181. Resumable `rows.jsonl`; `analyze.py` gives the paired sign test.

## Prediction (written session #16, before any run; smoke rows on positions 3900–3901 were discarded)
- **Validity check:** greedy CoT `\boxed{}` rate ≥ 85% and accuracy 25–60%. Otherwise the verdict is marked compromised.
- **H1 (confirmatory):** G3 − greedy CoT ≥ +3.0 AND sign-test p < 0.05 at n=250. Predicted **+4** (range −1 to +9), ~45%.
- Reported: PAL single-sample accuracy vs greedy CoT; per-level diffs; token cost.
- No verdict before n=250 (expect two lab runs; ~60–70 s/q on the lab machine).

## Results
(pending)
