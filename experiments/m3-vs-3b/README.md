# m3-vs-3b: can label-free self-training let 1.5B match the 2× bigger model, where inference tricks could not?

**Why.** Two confirmatory results (math-g3-xscale −5.5 on MATH; gsm8k-xscale −6.0, p=0.044 on GSM8K) say that at about
matched inference compute, Qwen2.5-3B greedy beats Qwen2.5-1.5B + G3. Inference tricks don't substitute for parameters.
Training is a one-time cost; if label-free self-training (m3-dpo3 MIXR: cross-format CoT+PAL vote → DPO, +5.3) brings the
1.5B *greedy* model level with 3B greedy, then training compute can substitute for half the parameters at inference.
This is the cheapest test possible: m3-dpo3 already stored per-question greedy results of BASE-1.5B and MIXR-1.5B on
1500 held-out GSM8K-train questions + 250 BBH date (drift). Only 3B greedy on the same items is missing.

## Design
- Kaggle T4, `run.py` = byte-identical copy of `m3-dpo3/run.py`; only `--phase eval --model Qwen/Qwen2.5-3B-Instruct`
  with no LoRA dirs → evaluates 3B "BASE" only. Same prompts (SYS_COT, BBH_COT), same held-out idx (shuffle seed 2029,
  [4500:6000]), same vLLM fp16 backend, greedy, max_new 512, same scorer. Output `gpu-output/eval_BASE.jsonl` (= 3B).
- Compare in-session with `compare.py` against `m3-dpo3/gpu-output/eval_{BASE,MIXR}.jsonl` (paired by idx, sign test).
- Caveat: one MIXR training draw (m3-dpo2 gave +4.2 on a different held-out set; same recipe). Same-runtime comparison
  (vLLM fp16 both), so no quantisation confound.

## Prediction (written session #22, before any run)
- **Validity:** 3B output has 1500 GSM8K + 250 BBH rows; GSM8K accuracy of 3B in 82–93% (1.5B BASE 81.7).
- **H1 (non-inferiority, confirmatory):** acc(MIXR-1.5B) ≥ acc(3B greedy) − 2.0 on the 1500 GSM8K items.
  Predicted 3B ≈ 87.0 vs MIXR 87.1 → **+0.1**, pass (~55%).
- **H2 (superiority, confirmatory):** MIXR − 3B ≥ +2 AND sign-test p < 0.05. Predicted **fail**.
- **H3 (generality, confirmatory):** on BBH date (untrained skill), 3B − MIXR ≥ +5 points. Predicted **pass** (3B ≈ 60 vs 48):
  the self-trained gain is skill-specific; the bigger model's advantage remains elsewhere.
- Exploratory: fraction of the BASE→3B gap closed by MIXR; output length (tokens) per arm.

## Results (session #23, Kaggle run 2026-11-14, 51 GPU min) — confirmatory
- **Validity PASS:** 1500 GSM8K + 250 BBH rows; 3B GSM8K acc 92.3 (range 82–93).
- GSM8K held-out (n=1500): 3B greedy 92.3, MIXR-1.5B 87.1, BASE-1.5B 81.7.
- **H1 FAIL:** MIXR − 3B = −5.3 (fixed 52 / broken 131), p=4.7e-9. Predicted +0.1 (non-inferiority margin −2).
- **H2 FAIL** (as predicted).
- **H3 PASS:** BBH date 3B 60.4 vs MIXR 48.0: +12.4 (60/29), p=0.0013 (BASE 50.0; MIXR−BASE −2.0, p=0.66).
- Exploratory: MIXR closes 50% of the BASE→3B gap on its trained skill (3B−BASE +10.6) and 0% elsewhere.
  3B outputs are longer (876 vs 720 chars).
- Reading: label-free DPO, like G3 at inference, closes about half of the 1.5B→3B gap — not all of it — and only on
  the trained skill. Prediction was wrong because I anchored on 3B's GSM8K *test* accuracy (~85) instead of the easier
  held-out *train* items (BASE 81.7 there vs ~68 on test). Open: do training and G3 stack to close the full gap?

## Review 2026-11-15 (session #24)
Not reworded: the claim is relative to items drawn from GSM8K train, which look contaminated (BASE 81.7 vs 69.8 on test). "Closes 50% of the gap" holds on these items only. m3-stack adds 3B and MIXR greedy on the full GSM8K test. compare.py reads only gpu-output/ (and m3-dpo3/gpu-output), confirmed.
