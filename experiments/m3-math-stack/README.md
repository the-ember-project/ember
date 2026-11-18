# m3-math-stack: does label-free DPO on MATH survive a 2048-token eval, a fresh data draw, and stack with G3 toward 3B?

**Why.** m3-math (n=1200) gave label-free DPO +3.75 (MIXR) / +4.33 (COT-majority) on public MATH test, but validity
failed: BASE left 13.4% unboxed at a 1024-token cap, and ~half of each gain was learning to *finish and box*.
It was also one training run per arm (same-format DPO gave +3.1 and −7.9 on two GSM8K draws). This job fixes all
three at once and adds the M5 test on MATH:
1. Eval cap 2048 tokens (truncation confound mostly gone).
2. Fresh training draw (MATH-train positions 2000–3999 of the seed-2029 shuffle; m3-math used 0–1999) and fresh test
   items (MATH-test positions 1200–2199; m3-math used 0–1199).
3. G3 (greedy CoT + 3 PAL at T=0.7, plurality vote) for BASE and the trained model, and Qwen2.5-3B greedy on the same
   items with the same code → candidate M5 ("1.5B + training + tricks ≥ 3B greedy − 2").

Only the **COT-majority** arm is trained (label = ≥5 of 8 CoT answers agree): on MATH it matched MIXR (−0.6, p=0.65)
with equal label precision, and it needs no PAL samples, which saves ~40 GPU min. This is the TTRL/ScPO-style label.

## Design
Copy of `../m3-math/run.py` (same recipe: Qwen2.5-1.5B-Instruct fp16, vLLM gen greedy + 7 CoT at T=0.8, max 1024
gen tokens, LoRA r=16 DPO+NLL β=0.1 α=1 lr 5e-5 1 epoch, ≤2 pairs/question, cap 1200, DPO max_len 1200, vLLM seed
6061) with these changes: ARMS=["COT"], no PAL gen, `--train_off 2000`, `--test_off 1200 --n_test 1000`, greedy eval
at `--eval_max_new 2048` (vLLM max_model_len = max_new+640), G3 phase from `../m3-stack` (PAL max 1024 tokens),
then 3B greedy eval (tag `3b_`, no BBH). BBH date (n=250) drift for BASE vs COT as before. Every phase is smoke-tested
on a tiny slice inside the Kaggle job first. CPU smoke (0.5B, HF) of all five steps: see Runs.

## Prediction (written session #26, before any run)
- **Validity:** BASE greedy unboxed rate ≤ 5% at 2048 tokens; BASE accuracy 45–68%; 3B greedy accuracy ≥ BASE + 3;
  PAL exec rate ≥ 55% (math-mix7 interim 63%). If any fails, verdicts are reported but marked compromised.
- **H1 (M3 on MATH without the truncation confound, fresh draw; confirmatory):** COT − BASE greedy ≥ +2.0 AND
  sign-test p < 0.05 (n=1000). Predicted **+2.0** (range −1 to +5), ~50%: in m3-math the boxed-items-only gain was
  +2.25 (p=0.04) for COT, and a larger cap removes most of the finishing advantage.
- **H2 (M5 non-inferiority; confirmatory):** G3(COT) ≥ 3B greedy − 2.0 (point estimate, paired on the same 1000).
  Predicted **−4** (BASE ~57, COT ~59, G3(COT) ~62, 3B ~66) → FAIL (~25% pass).
- **H3 (G3 on the trained model; confirmatory):** G3(COT) − COT greedy ≥ +2.0 AND p < 0.05. Predicted **+3** (~65%).
- **H4 (drift):** COT − BASE on BBH date ≥ −3 (point estimate). Predicted −2 (~55%; m3-math COT arm had −5.6).
- Reported, no criterion: G3(COT) − G3(BASE); BASE/COT unboxed rates; tok/q; compute ratio (tokens × params) of
  G3(COT) vs 3B greedy; label precision; per-level diffs; fixes on BASE-unboxed vs boxed items.
- Verdicts computed from `gpu-output/*eval_*.jsonl` and `g3_*.jsonl.gz` (per-question), not only summary files.

## Runs
- CPU smoke (session #26, 0.5B, HF, tiny caps): gen → train (0 pairs at this size, as in m3-math/m3-stack) → eval →
  g3 → 3B-tagged eval (n_bbh 0) all exit 0; eval + g3 re-run with a dummy LoRA in `lora_COT` exercises the adapter
  path; `compare.py /tmp/...` runs on the outputs. Queued on Kaggle behind m3-stack: `run.py 250` (estimate ~190 min:
  gen ~80, train ~33, eval ~20, g3 ~15, 3B ~15, setup+smoke ~25). Phases save as they go.

## Results (session #29, Kaggle run 1, 204 GPU min, vLLM; confirmatory, **validity compromised on 2 of 4 checks**)
Fresh MATH-test positions 1200–2199 (n=1000), greedy at 2048 tokens. COT-majority DPO trained on 1067 pairs
(677 questions, label precision 90.8%).
- **Validity:** BASE unboxed 7.4% (need ≤5: FAIL); BASE acc 55.5 (45–68: PASS); 3B 68.6 ≥ BASE+3 (PASS);
  PAL exec rate 46.2% pooled (need ≥55: FAIL; BASE 58.3%, COT 34.2%). Verdicts below are reported but compromised.
- Accuracy: BASE 55.5, COT 56.3, G3(BASE) 59.3, G3(COT) 58.6, 3B greedy 68.6.
- **H1 FAIL:** COT − BASE = +0.8 (82 fixed / 74 broken), p=0.58. Predicted +2.0. On BASE-boxed items: +0.1 (72/71);
  on BASE-unboxed items: +9.5 (10/3, n=74). **The m3-math DPO gain on MATH does not survive a fresh draw at 2048
  tokens; what remains is finishing/boxing on the few truncated items.**
- **H2 FAIL:** G3(COT) − 3B greedy = −10.0 (60/160), p=1e-11. Predicted −4.
- **H3 PASS:** G3(COT) − COT = +2.3 (31/8), p=0.0003. Predicted +3.
- **H4 PASS:** BBH-date drift COT − BASE = +2.4 (41/35). Predicted −2.
- Exploratory: G3(COT) − G3(BASE) = −0.7 (n.s.). DPO on CoT traces **cut the model's PAL exec rate from 58% to 34%**
  and PAL-1 accuracy from 29.6% to 18.1%; PAL tokens/q rose 816 → 1316 (programs got longer / ran on). So G3 on the
  trained model gains less per voter. 3B − BASE = +13.1 (177/46). Per level, COT − BASE is within ±3.5 everywhere.
- **Reading:** label-free COT-majority DPO at 1.5B does not improve MATH reasoning when the eval isn't truncated;
  M3-on-MATH (m3-math) should be read as mostly finishing/boxing. Training on one format damaged the other format.
