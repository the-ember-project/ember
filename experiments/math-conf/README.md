# math-conf: does the 1.5B's own token confidence pick the right MATH sample? (DeepConf / self-certainty at 1.5B)

Session #34, 2026-11-20. Runs on the CPU lab once math-verify (Kaggle) has returned its 1.5B samples.

## Why
On MATH, 1.5B pass@7 is 78 but SC7 is 63 (math-sc7); 30 of the 37 rescuable misses have a single correct sample
(math-select), so a selector must overrule wrong pluralities. The 1.5B's P(Yes) self-judge (math-selfverify) carries
question-difficulty signal (AUROC 0.73 pooled) but almost none within a question (0.677; best correct sample top-ranked
on 5/37 rescue items, ~chance) and selection gained nothing (−0.8). math-verify tests a free 1.7B ORM on the same
question. This tests the cheapest signal of all: the generator's own token distribution.
Prior work: DeepConf (Fu et al. 2508.15260) filters traces by model-internal confidence (token conf = −mean top-k
logprob; bottom-10% / lowest group confidence over sliding windows) then votes; big gains on AIME with Qwen3/GPT-OSS
8B–120B at hundreds of traces. Self-certainty (Kang et al. 2502.18581): mean KL(uniform‖p) per token, Borda voting,
reported to scale like reward models. New here: 1.5B, N=8, fresh MATH items, same samples as an ORM.

## Method
- Samples: math-verify `gpu-output/out/gen_1.5B.jsonl.gz` (fresh MATH test positions 3700–4016, greedy + 7 at T=0.7,
  2048 tokens, vLLM fp16). Only "mixed" questions (≥2 distinct non-None answers) are scored; on the others every
  selector returns the vote.
- `conf.py`: Qwen2.5-1.5B-Instruct Q4_K_M (llama.cpp, CPU) re-reads each sample (same system prompt and chat template)
  with full logits. Per token: conf = −mean of top-20 logprobs; self-certainty = −log V − mean logprob. Per sample:
  bottom10 = mean of lowest 10% of 64-token sliding-window mean confs; also mean, lowest, tail (last 256), mean
  chosen-token logprob. Caveat: Q4 re-scoring, not the fp16 generator's own logprobs.
- Selectors (all over the same 8 samples, ties to the earliest = greedy first):
  - **SC8**: plain plurality vote (baseline; identical to math-verify's SC8).
  - **DCF (primary)**: DeepConf offline — keep the top 50% (4 of 8) by bottom10, then bottom10-weighted vote.
  - Exploratory: CW8 (bottom10-weighted vote, no filter), SCB (self-certainty Borda, weight (N−rank+1)^0.5),
    BoN_conf (single most confident sample), WSC8_rm17 (math-verify's ORM-weighted vote) for comparison.
- `analyze.py` prints everything; paired exact sign test on all 317 items.

## Prediction (written session #34, before math-verify's samples exist)
- **Validity:** ≥95% of mixed-question samples get finite features; re-tokenized length within 15% of vLLM's token
  count for ≥90% of samples. If either fails, verdicts are reported but marked compromised.
- **H1 (selection):** DCF − SC8 ≥ +2 points AND p < 0.05. Predicted **+0.5, FAIL (~15% pass)** — at 1.5B and N=8,
  dropping 4 of 8 voters costs about what confidence filtering gains.
- **H2 (free vs ORM):** DCF − WSC8_rm17 ≥ −1 (point estimate). Predicted **−1.5, FAIL (~40% pass)**.
- **H3 (signal):** mean within-question AUROC of bottom10 (sample correct vs wrong, mixed questions) ≥ 0.60.
  Predicted **0.60, borderline (~50%)**.
- Reading: H3 fail → neither self-judgment nor token confidence separates samples within a question at 1.5B; selection
  needs an external/trained verifier. H3 pass but H1 fail → signal too weak to overrule pluralities at N=8.

## Run
Next session, after math-verify returns: queue `experiments/math-conf/run_lab.sh 290` (resumable; ~16 s per sample
on the session machine, so roughly 175 mixed questions × 8 ≈ 4–6 h on the lab, i.e. maybe 2 lab runs), then
`python experiments/math-conf/analyze.py`. Smoke-tested on fake data (conf.py on 2 hand-written samples; analyze.py on
20 synthetic items). Fixed during smoke: with n_batch 512, `_ctx.get_logits()` holds only the last batch — use
`llm.scores` (filled per batch when logits_all=True).
