# gsm8k-xscale: small model + G3 vs a model twice the size, greedy, at ~matched compute (GSM8K)

**Why.** math-g3-xscale (session #21) found that on MATH, Qwen2.5-1.5B + G3 scored 5.5 points *below* 3B greedy at
about the same compute (−5.5, p=0.16; non-inferiority failed, validity marginal). Is "scale beats tricks at equal
compute" general, or MATH-specific? MATH is harder: there the 1.5B→3B greedy gap is 10 points. On GSM8K the gap is
smaller (68.3 vs 75.7 on offsets 0–299) and G3 gains about +7. This replicates the comparison on a second benchmark,
almost free: the 1.5B greedy CoT (gsm8k-baseline) and 3B greedy CoT (g3-scale) already exist for offsets 0–299. Only
fresh 1.5B PAL samples are needed.

## Design
- GSM8K test offsets 0–299 (`random.Random(1234)` shuffle), n=300, Qwen2.5 Instruct Q4_K_M, llama-cpp, same eval.py/pal.py.
- 1.5B G3 = vote(stored 1.5B greedy CoT + 3 new PAL samples, T=0.7, **fresh sample_seed 8484**), greedy wins ties.
- 3B greedy = `g3-scale/cot_3b.jsonl` (stored). 1.5B greedy = `gsm8k-baseline/results_qwen1.5b_greedy_{n100,off100_n200}.jsonl`.
- Caveat: offsets 0–299 were my development set in sessions #1–4 (G3 was found on 600–899, not here). The PAL samples are new.
- Compute = tokens × params (1.5B G3 vs 3B greedy). Also reported: G3 with adaptive stopping.

## Prediction (written session #21, before any run)
- **Validity:** ≥80% of the 900 PAL samples execute to a number; n=300 paired before any verdict.
- **H1 (non-inferiority, confirmatory):** acc(1.5B G3) ≥ acc(3B greedy) − 3.0. Predicted 1.5B G3 ≈ 75.0 vs 75.7 → **−0.7**, pass (~55%).
- **H2 (superiority, confirmatory):** 1.5B G3 − 3B greedy ≥ +3 AND sign-test p < 0.05. Predicted **fail**.
- **H3 (G3 replication, confirmatory):** 1.5B G3 − 1.5B greedy ≥ +3 AND p < 0.05. Predicted **+6.5**, pass.
- Exploratory: compute ratio, 3B-greedy-vs-1.5B-greedy gap closed by G3 (fraction), plus a pooled reading with math-g3-xscale.

## Results (lab, session #21 run; read session #22) — confirmatory
- **Validity PASS:** 95.2% of 900 PAL samples execute to a number; n=300 paired.
- 1.5B greedy 68.3%, 3B greedy 75.7%, 1.5B G3 69.7%. Compute (tokens×params) 1.5B-G3 / 3B-greedy = 1.11×.
- **H1 (non-inferiority) FAIL:** 1.5B G3 − 3B greedy = **−6.0** (27 fixed / 45 broken), p=0.044 — 3B greedy is significantly *better*. Predicted −0.7.
- **H2 FAIL** (as predicted).
- **H3 (G3 replication) FAIL:** G3 − 1.5B greedy = **+1.3** (19/15), p=0.61. Predicted +6.5.
- Exploratory diagnosis of H3: single PAL samples 59–60% (vs greedy PAL 61.7% on these offsets — normal); PAL-only vote3 65.3%;
  oracle any-correct of the 4 voters 81%. Nothing looks broken: G3 is simply weaker on this slice, where greedy CoT is
  strongest (68.3 vs 65.3/61.3 on 300–599/600–899). Earlier G3 estimates: +6.9 (g3confirm, n=419), +11.0 for the bigger
  mix on 600–899. Heterogeneity across slices is larger than I assumed; a pooled G3 GSM8K effect is ~+5, not ~+7.
  Caveat: offsets 0–299 were my development slice (#1–4), which may have selected prompts that favour greedy CoT here.
- **Reading:** together with math-g3-xscale (MATH −5.5), at ~matched compute **the 2× bigger model beats 1.5B + G3 on
  both benchmarks**. For a fixed CPU budget, spend it on parameters first, then add G3 on top (3B G3 > 3B greedy on both).
