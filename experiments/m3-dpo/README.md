# m3-dpo: label-free preference training, with the CoT+PAL vote picking the preferred rationale

Roadmap M3, run 3. In m3-selfdistill (run 2), SFT on the model's own rationales gave nothing, even with gold filtering (MIX 0.0, GOLD −0.5, n=419). Diagnosis: the model already rates its own rationales as highly likely (loss ~0.15), and only ~10% of examples correct an error. Preference training puts its signal on exactly the questions where samples disagree. **ScPO** (Prasad et al., ICML'25, arXiv:2411.04109) did this with self-consistency votes and roughly matched gold-label IRPO on GSM8K for Llama-3-8B. **What's new here:** the pseudo-label comes from the cross-format CoT+PAL vote (on run-1/2 data, its precision on questions with disagreement among the 5 CoT samples is 91.9% vs 84.4% for CoT-majority), on a 1.5B model, with a direct comparison against the CoT-only (ScPO-like) label and the gold upper bound.

## Design
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, Kaggle T4, vLLM for generation/eval (HF fallback).
- Train pool: GSM8K **train**, `random.Random(2029)` shuffle, first 3000 questions. This is a new shuffle, so the set partly overlaps run 2's questions; that's fine because it's train data.
- Per question: greedy CoT + 7 CoT samples (T=0.8), and greedy PAL + 4 PAL samples (T=0.8). Prompts and scorer are the same as in m3-selfdistill.
- Labels:
  - **MIX** (no gold): the plurality of all 13 answers. Kept if it has ≥7/13 votes and ≥1 CoT and ≥1 PAL vote.
  - **COT** (no gold, ScPO-like): the plurality of the 8 CoT answers. Kept if ≥5/8.
  - **GOLD** (upper bound, IRPO-like): the gold answer.
- Pairs: for each kept question, chosen = a distinct CoT text whose answer equals the label, and rejected = a distinct CoT text with a different or unparsed answer. Up to 2 pairs per question. All arms are subsampled to the same E pairs (≤2000).
- Training: LoRA r=16, α=32, all projections. Loss = DPO (β=0.1, reference = the base model with the adapter disabled) + 1.0 × length-normalised NLL on the chosen text (the RPO/IRPO recipe). 1 epoch, lr 5e-5, batch 16 pairs, warmup 10%, linear decay. Logged: DPO loss, NLL, pair accuracy, log-ratio margin.
- Eval: greedy CoT on the **full GSM8K test set (1319 q)**. Earlier fp16 work never trained or tuned on it; run 2's eval saw offsets 900–1318 but nothing was selected on them. Drift check: greedy CoT on BBH date_understanding (250 q, the bbh-mixvote2 prompt and parser), for every arm. Paired exact sign tests.
- In-job smoke pass of all phases (16 train q, 4 test q) before the full run.

## Prediction (written before any data from this run exists)
- **H1 (primary, confirmatory):** MIX − BASE on GSM8K test (n=1319). **Pass: ≥ +2.0 points AND sign-test p < 0.05.** Predicted **+2.5** (range −1 to +6). I give a full pass about 35%: run 2 showed this model is hard to move, but DPO targets the errors directly.
- **H2:** MIX − COT ≥ 0. Predicted +0.5. The two arms' labels differ on only ~3% of questions, so this will almost surely not be significant. Reported with p.
- **H3 (drift):** MIX − BASE on BBH date_understanding ≥ −3 points (no meaningful harm to a non-GSM8K skill). Predicted ~0.
- Reference (not pass/fail): GOLD − BASE predicted +3; COT − BASE predicted +2; BASE GSM8K greedy predicted 70% (range 66–74).
- Train pairs: MIX label precision on pair questions ≥ COT's. Predicted ~92 vs ~85.
- Anything else is exploratory.

## Runs
GPU job `run.py` via self/gpu_queue.txt (limit 330 min; expected ~3.5 h with vLLM). Outputs: `summary.json`, `train_stats.json`, `train_log.json`, `eval_<ARM>.jsonl` (GSM8K and BBH tails), `gen_answers.jsonl.gz`.

## Results (session #9; Kaggle, 166 GPU min, vLLM)
Training learned in every arm (`train_log.json`): DPO loss 0.694 → 0.52–0.59, reward margin ~0 → 3.4–5.1, pair accuracy 0.33 → 0.68–0.83 over 108 steps, E=1727 pairs per arm.

GSM8K test, full 1319, greedy (`summary.json`):
| arm | acc | vs BASE | fixed/broken | p |
|---|---|---|---|---|
| BASE | 69.8% | — | — | — |
| MIX (CoT+PAL vote ≥7/13) | 72.2% | +2.35 | 155/124 | 0.072 |
| COT (CoT majority ≥5/8, ScPO-like) | 72.9% | +3.11 | 145/104 | 0.011 |
| GOLD | 75.4% | +5.53 | 164/91 | <0.001 |

BBH date_understanding drift (n=250): BASE 50.0, MIX 48.0 (−2.0, p=0.64), COT 47.2 (−2.8, p=0.48), GOLD 45.2 (−4.8, p=0.21).

Pre-registered criteria, each as measured (confirmatory):
- **H1 FAIL:** MIX−BASE +2.35 meets the +2 threshold but p=0.072 > 0.05. Close to the predicted +2.5, but not distinguishable from noise at this n.
- **H2 FAIL:** MIX−COT −0.76 (111/121, p=0.55). No difference detectable.
- **H3 PASS:** MIX BBH drift −2.0 ≥ −3 (all drift estimates are noisy, n=250; GOLD's −4.8 would fail the same bar).
- Pair-label precision PASS: MIX 95.4% vs COT 89.6% on pair questions (predicted ~92 vs ~85).
- Reference arms: BASE 69.8% (predicted 70); GOLD +5.5 (predicted +3); COT +3.1 (predicted +2).

Interpretation:
- **DPO moves this model where SFT did not** (m3-selfdistill: GOLD SFT −0.5; here GOLD DPO +5.5, p<0.001). Negative examples are the missing signal.
- A **label-free** arm (COT, ScPO-like) gave +3.1, p=0.011 — this meets the M3 bar (≥+3) numerically, but it was a reference arm, so it is not the confirmed primary; it needs a fresh test.
- Better pseudo-labels (MIX) did not translate into a bigger gain. Exploratory (`pair_difficulty.py` → `pair_difficulty.txt`): MIX and COT pair questions are equally easy (mean 5.8/8 CoT samples correct; only 15.5% / 10.4% "hard" with ≤4/8 correct), while GOLD pairs are 34.6% hard. GOLD's extra gain likely comes from hard questions where the model's majority is wrong — which a strict vote filter excludes. A relaxed MIX (plurality ≥5/13, both formats) keeps 1301 pair questions, 93.2% precision, 24.7% hard — better than COT on both precision and hard coverage. Candidate label for a next round.
- GOLD shows the largest BBH-date drop (−4.8, n.s.); watch drift in later rounds.

## Addendum (session #10, after weekly review 2026-11-08)
- Drift per arm (BBH date, n=250): MIX −2.0 (p=0.64), COT −2.8 (p=0.48), **GOLD −4.8** (p=0.21). The −4.8 arm is GOLD (reference), so the label-free COT arm stayed within the −3 tolerance here.
- **Relabel:** "DPO moves the model where SFT did not; negative examples were the missing signal" is an *interpretation, confounded*: DPO pairs come only from disagreement (harder) questions, the recipe differs (NLL term, lr, pairs, epochs) and the eval sets differ (n=1319 vs 419). Data selection could explain the GOLD gain as well as negatives. Control needed: NLL-only on the chosen side of the same GOLD pairs. Prior work anticipating the negatives reading: Setlur et al. 2024 (arXiv:2406.14532). The COT-majority arm is essentially a ScPO (arXiv:2411.04109) replication at 1.5B.
- The MIX label-precision advantage partly reflects keeping easier questions (pair_difficulty.py: 15.5% hard vs 34.6% for GOLD).
- m3-dpo2 (fresh gen seed, held-out train eval): COT-majority DPO **−7.9** (p<0.001), so the +3.1 here did not replicate; label-free DPO is unstable across data draws.
