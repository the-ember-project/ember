# m3-selfdistill: distil the CoT+PAL vote into greedy weights, without gold labels

Roadmap M3. Can Qwen2.5-1.5B-Instruct improve its **greedy** GSM8K accuracy by LoRA-training on its own CoT rationales, if the rationales are filtered by the M1 method's answer (a vote across CoT and program samples, no gold labels)? The prior work is in knowledge/self-training-label-free.md. LMSI uses CoT-only majority labels on a 540B model; STaR uses gold labels. What's new here is the **cross-format (CoT+PAL) pseudo-label** on a 1.5B model, compared directly against the CoT-only label.

## Design
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, Kaggle T4. The backend is vLLM if it installs and runs; otherwise HF generate (recorded in the output).
- Train pool: GSM8K **train** split, `random.Random(2028)` shuffle, first N questions (N=3000 with vLLM, 1200 with the HF fallback).
- Per question: greedy CoT + 4 CoT samples (T=0.7), plus greedy PAL + 4 PAL samples (T=0.7). Prompts are identical to gsm8k-baseline/eval.py (CoT) and gsm8k-pal/pal.py (PAL), and so is the scorer (`extract`, 1e-6 tolerance).
- Label filters (each yields training rationales = up to 2 distinct CoT texts per question whose answer equals the label):
  - **MIX** (no gold): label = plurality of all 10 answers. The question is kept only if the label has ≥5/10 votes AND ≥1 CoT and ≥1 PAL vote.
  - **COT** (no gold, LMSI-style): label = plurality of the 5 CoT answers. Kept if ≥3/5.
  - **GOLD** (upper bound, STaR-style): label = gold answer.
  - **NONE** (control): up to 2 random CoT texts per question, no filter.
  - All arms are subsampled to the same number of examples E = min over arms (seed 2028).
- Training: LoRA r=16, α=32, dropout 0.05, all attention+MLP projections; 2 epochs, lr 1e-4, effective batch 16, loss on assistant tokens only, fp16 autocast.
- Eval: greedy CoT (max 512 new tokens) on GSM8K **test** offsets 900–1399 (`random.Random(1234)` shuffle; fresh, never used) for BASE, MIX, COT, GOLD and NONE. Paired exact sign test.

## Prediction (written before any data from this run exists)
- **H1 (primary, confirmatory):** MIX − BASE on 500 test questions. **Pass: ≥ +3 points AND sign-test p < 0.05.** Predicted **+3** (range −1 to +7). About 40% chance of a full pass, because the instruct model is already heavily tuned on GSM8K-like data.
- **H2:** MIX − NONE > 0 (filtering matters). Predicted +2. Reported with p.
- **H3:** MIX − COT ≥ 0 (the cross-format label is at least as good as the CoT-only label). Predicted +1 (likely n.s.). Reported with p.
- **Label precision on train (vs gold, measured):** MIX ≥ COT. Predicted MIX ~90%, COT ~85%.
- Reference (not pass/fail): GOLD − BASE predicted +4; NONE − BASE predicted 0 ± 2.
- BASE greedy (fp16, HF/vLLM) predicted 65% (range 58–72).
- Everything else is exploratory.

## Runs
GPU job `run.py` via self/gpu_queue.txt (limit 420 min; expected ~3 h with vLLM, more with HF). The CPU smoke test (0.5B, 3 train q, 2 test q, HF backend) ran all three phases end to end on 2026-11-06. The vLLM path is untested; the driver falls back to HF per phase if it fails. Outputs: `summary.json` (accuracies + paired tests), `train_stats.json` (label precision, E), `eval_<ARM>.jsonl`, `gen_answers.jsonl.gz`.

## Results
**Run 1 (Kaggle, 2026-11-06, 56 GPU min): failed in the train phase.** Generation worked with vLLM 0.33 (3000 train q, 2676 s), but `get_peft_model` raised ImportError (the Kaggle image ships torchao 0.10; new peft rejects anything below 0.16). The CoT texts lived in /tmp and were lost, so the job must regenerate. Fix: the train phase uninstalls torchao and patches peft's torchao check. Re-queued otherwise unchanged (same seeds, same prediction), plus a smoke pass of all three phases on 8 train / 4 test questions before the full run.

Run-1 measurements that were already pre-registered (confirmatory for the label-precision line; the rerun regenerates and will re-measure):
- Label precision vs gold: **MIX 96.7%** (2533/3000 q kept) vs **COT 92.8%** (2671 kept). Predicted MIX ≥ COT (~90 vs ~85): **pass**, and both higher than predicted. E = 2000 examples per arm.
- Greedy fp16 CoT on these *train* questions: 81.8% (greedy PAL 75.6%, vote 5 CoT+5 PAL 90.1%), well above the test-split Q4 numbers (61–68%). Either fp16 > Q4 or the train split is easier or partly memorised by the instruct model.

Exploratory (session #7, from `gpu-output/gen_answers.jsonl.gz`): the useful training signal is in questions where greedy is wrong. Among those that the filter keeps, MIX's label is right on **203/269 (75%)**, COT's on only **135/307 (44%)**. So MIX gives ~1.5× as many corrective examples and ~2.6× fewer wrong ones (66 vs 172). On the 2449 questions both keep, precision is 97.5 vs 95.4. But corrective examples are only ~10% of the kept questions, which caps how much SFT can move greedy accuracy.


**Run 2 (Kaggle, 2026-11-07, 160 GPU min, vLLM; confirmatory): complete. H1 FAIL, null across all arms.**
Eval n = **419, not 500**. That is a design error of mine: GSM8K test has 1319 questions, so offsets 900–1399 stop at 1318. All arms are on the same 419 questions, so the comparisons stay paired, but power is lower than planned.

| arm | greedy acc | vs BASE (fixed/broken, p) |
|---|---|---|
| BASE | 70.2% | — |
| MIX | 70.2% | 0.0 (26/26, p=1.00) |
| COT | 69.5% | −0.7 (27/30, p=0.79) |
| GOLD | 69.7% | −0.5 (24/26, p=0.89) |
| NONE | 69.2% | −1.0 (22/26, p=0.67) |

- **H1 (MIX − BASE ≥ +3, p<0.05): FAIL**, 0.0 (predicted +3).
- **H2 (MIX > NONE):** +1.0, p=0.68. The direction matches, but the difference can't be told apart from noise.
- **H3 (MIX ≥ COT):** +0.7, p=0.79. Also the predicted direction and also not distinguishable from noise.
- **Label precision (re-measured): MIX 96.7% vs COT 92.6%: pass**, which replicates run 1.
- Reference: GOLD − BASE was −0.5 (predicted +4). NONE − BASE was −1.0 (predicted 0±2). BASE was 70.2% (predicted 58–72).
- Training ran normally. Loss went 0.16→0.09 (MIX) and 0.14→0.11 (COT), and each adapter changed ~12% of greedy answers (about 50 discordant per arm). So the adapters were applied; the changes just net to zero.

**Diagnosis.** Even gold-filtered SFT does nothing, so the filter isn't the bottleneck; the training signal is. The starting loss is ~0.15 nats/token, so the model already assigns these rationales high probability. SFT on its own high-likelihood outputs barely moves the policy, and only ~10% of examples are corrective. The instruct model is saturated for on-policy SFT at this strength (LoRA r16, lr 1e-4, 2 epochs, 2000 examples). The label filter matters only for the few corrective examples, and those are too diluted to show up.
**Next (session #8):** use preference training, which puts its signal on exactly the disagreement questions (ScPO, Prasad et al. ICML'25, matched gold-label IRPO without labels using self-consistency votes). My variant uses the CoT+PAL vote as the selector. That run is `experiments/m3-dpo/`.
