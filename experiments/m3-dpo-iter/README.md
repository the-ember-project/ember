# m3-dpo-iter: does label-free MIXR-DPO compound over a second round?

M3 is met on GSM8K (m3-dpo-rep, #45): one round of label-free MIXR-DPO (chosen/rejected CoT picked by the model's
own CoT+PAL vote, no gold) gives +3.8 greedy on the full GSM8K test (3 seeds, +2.9/+3.3/+5.3). The obvious next
question for a self-improvement loop: **does a second round, with samples and labels from the round-1 model, add
more, or does it saturate?** On gridwalk iteration compounded (m3-gw-iter, SFT: R2 74.0 vs R1 68.8).
Prior work (from memory, numbers unverified): Iterative RPO (Pang et al. 2404.19733) compounds over 3–4 rounds with
*gold* labels on GSM8K; ScPO (Prasad et al. 2411.04109) runs 2 self-consistency-labelled rounds, with a smaller
second-round gain. Open here: label-free, 1.5B instruct, CoT+PAL-vote labels, uncontaminated test.
A risk specific to self-labels: after round 1 the model agrees with itself more, so fewer questions have both a
labelled and a non-labelled sample (fewer pairs), and its PAL may degrade (DPO cut PAL exec on MATH, m3-math-stack).

## Design
- Two independent chains, one per T4: **A** (gen seeds 7071/7072, train seed 3041), **B** (8081/8082, 3042).
- Round 1: MIXR recipe from m3-dpo-rep (k_cot 7, k_pal 4, T=0.8, β 0.1, α 1, LoRA r16, 1 epoch, pairs capped as
  before) on GSM8K-train questions 0–1999 of the seed-2029 shuffle (**2000 questions, not 3000**, to fit the GPU quota).
- Merge round-1 LoRA into the weights (fp32 on CPU, saved fp16) → R1m.
- Round 2: same recipe with R1m as policy, sampler and DPO reference, on **fresh** train questions 2000–3999.
- Eval (greedy CoT, full GSM8K test, n=1319): step r1 = original model: BASE and R1 (LoRA); step r2 = R1m (merged,
  "BASE" in that file) and R2 (LoRA on R1m). R1 is evaluated before round 2 starts.
- `iter.py` (driver: smoke chain on 16 questions first, then full chain; one chain per GPU), `run.py` (m3-dpo-rep's,
  plus `--q_off` and a `merge` phase), `analyze.py`.

## Prediction (written session #45, before any run)
Primary comparison is R2 vs R1m on the same test items (paired sign test), per chain.
- **I1 (compounding, decisive):** mean over chains of R2 − R1m ≥ +1.0 and both chains > 0. Predicted mean **+1.2 →
  PASS (~55%)**.
- **I2 (each round-2 gain significant):** R2 − R1m p < 0.05 in both chains. Predicted **FAIL** (~25%).
- **I3 (total):** mean over chains of R2 − BASE ≥ +5.0. Predicted R1 ≈ +3.3 (2000 q) + 1.2 ≈ **+4.5 → FAIL** (~40%).
- **I4 (round-1 replication at 2000 q):** each chain R1 − BASE > 0 and mean ≥ +2.0. Predicted PASS (+3.3).
- Validity: BASE 67–72; |R1m − R1(LoRA)| ≤ 1.0 pt (merge sanity); parsed-answer rate ≥ 99% for every model; round-2
  pairs ≥ 500 per chain (else round 2 is starved, report as such).
- Exploratory: pair counts and label precision per round (train_stats), PAL exec/agreement in round 2, response length.
- Reading: I1 PASS → the label-free loop compounds at least once at 1.5B; next is a 3rd round / data-matched control
  (one round on 4000 q) to separate iteration from data. I1 FAIL with pairs ≥500 → one round captures the gain.
