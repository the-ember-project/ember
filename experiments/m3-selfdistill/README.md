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
(pending)
