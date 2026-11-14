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

## Results
(pending)
