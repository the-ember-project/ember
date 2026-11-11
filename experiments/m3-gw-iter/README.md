# m3-gw-iter: does label-free self-training keep improving, and does it add on top of voting? (M3 follow-up)

**Background.** m3-gridwalk (session #15): SFT of Qwen2.5-1.5B on its own CoT traces that agree with the plurality
of its own PAL samples (PALV, label-free) raised greedy CoT on 10-step gridwalk 46.3 → 63.2 (+16.8, p<1e-20);
simulator-verified traces (GOLD) gave +23.8; same-format CoT-majority gave +1.0. Two open questions:
1. **Iteration (STaR/expert-iteration style):** if the round-1 model generates new traces and labels, does a round-2
   model improve further, or does label noise (13% wrong PALV labels) cap it?
2. **Compounding:** does self-training improve the best *cheap inference system* (G3 = vote(greedy CoT + 3 PAL)),
   or does it only move greedy CoT up toward what voting already achieved?

## Design (Kaggle T4; vLLM gen/eval with LoRA, HF+PEFT train; recipe identical to m3-gridwalk)
- Round 1: base model, train tasks `make_task(i,10,seed=9001)`, i<1500, 8 CoT + 5 PAL samples at T=0.8 (vLLM seed 5151;
  same as m3-gridwalk). PALV filter (PAL plurality ≥3/5; ≤2 distinct matching CoT traces per task), all traces kept.
  **R1** = LoRA (r=16, lr 1e-4, 2 epochs, eff. batch 16) from base on round-1 traces.
- Round 2: R1 model generates on *new* tasks `make_task(i,10,seed=9004)`, i<1500: 4 CoT + 5 PAL at T=0.8 (PAL from
  R1 too). Same PALV filter. **R2** = LoRA from base on round-1 ∪ round-2 traces (STaR restarts from base).
- Eval, fresh tasks: gw10 600 (`seed=9005`), gw14 300 (`seed=9006`), GSM8K test 300 (drift; same shuffle as before).
  Per arm (BASE, R1, R2): greedy CoT on all; on gw10 also 3 PAL samples at T=0.7 → **G3** = plurality of
  [greedy CoT, PAL1..3], ties to greedy CoT. Scoring: simulator, exact sign test, verdicts from per-item jsonl.
- Known confound: R2 is trained on ~2× the data of R1, so H1 can't separate "better teacher" from "more data".
  Secondary stats (round-2 label precision/coverage) speak to the teacher part.

## Prediction (written before any run)
- Sanity (not a hypothesis): BASE gw10 30–55%; skip rate (<30 tokens) ≤20%; R1 − BASE ≥ +10 (replication of +16.8).
- **H1 (iteration):** R2 − R1 on gw10 greedy CoT **≥ +3 AND p<0.05**. Predicted +5 (range −2 to +12); ~50%.
- **H2 (compounding):** G3(R2) − G3(BASE) on gw10 **≥ +3 AND p<0.05**. Predicted +5 (range −3 to +12); ~50%.
- **H3 (trained 1× vs base vote ~4×):** R2 greedy CoT ≥ G3(BASE) − 3 points (point estimate). Predicted R2 ≈ 68 vs
  G3(BASE) ≈ 72 → ~40%.
- **H4 (drift):** R2 GSM8K greedy ≥ BASE − 3 (point estimate). Predicted pass (~75%).
- Secondary, reported: round-2 PALV coverage and label precision vs round 1 (814/1500, 86.9%); precision of a JOINT
  rule (plurality of 5 PAL + 4 CoT answers with ≥5 votes); gw14 transfer for each arm; G3(R1).

## Results (session #16; CONFIRMATORY; Kaggle 244 GPU-min; verdicts recomputed from eval_*.jsonl, n=600 gw10)
Accuracy (greedy CoT / G3 on gw10; gw14 greedy; GSM8K greedy n=300):
| arm | gw10 | G3 gw10 | gw14 | GSM8K |
|---|---|---|---|---|
| BASE | 47.7 | 69.8 | 18.7 | 69.7 |
| R1 | 68.8 | 77.2 | 42.0 | 71.3 |
| R2 | **74.0** | **80.8** | 47.3 | 71.0 |
- Sanity: BASE 47.7 (in 30–55), skip rate 0% all arms; R1 − BASE +21.2 (139/12) — replicates m3-gridwalk (+16.8).
- **H1 PASS:** R2 − R1 = **+5.2** (38 fixed / 7 broken), p=3e-6. Predicted +5. Confound: R2 saw 2309 traces vs R1 1012.
- **H2 PASS:** G3(R2) − G3(BASE) = **+11.0** (103/37), p=2e-8. Predicted +5. G3 still adds on top of R2: +6.8 (83/42), p=3e-4.
- **H3 PASS:** R2 greedy 74.0 ≥ G3(BASE) − 3 = 66.8. In fact R2 greedy beats G3(BASE) by +4.2 (98/73, p=0.066):
  one trained greedy pass ≈ matches/exceeds the untrained 4-sample vote.
- **H4 PASS:** GSM8K drift R2 +1.3 (p=0.56), R1 +1.7.
- Secondary: round-2 PALV coverage rose (808→1113 of 1500 tasks) while label precision fell (87.3→83.0%); kept-trace
  correctness stayed 95.2/95.1%. The JOINT rule (5 PAL+4 CoT, ≥5 votes) went 81.3→90.7% precision on round 2, i.e. once
  CoT is trained it becomes a useful co-voter. gw14 transfer: R2 +28.7 vs BASE (90/4).
- Interpretation: label-free cross-format self-training iterates (round 2 adds) and compounds with voting (the
  self-trained model's G3 is the best system measured: 80.8%). Not separated: more data vs better teacher in round 2.
