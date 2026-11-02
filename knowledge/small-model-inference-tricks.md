# Inference-time tricks for small models (to test)

What I plan to use from each; results get linked here when measured.

- **Self-consistency** (Wang et al. 2022, arXiv:2203.11171): sample several CoT paths at T≈0.5–0.7, majority-vote the final answer. Big gains on GSM8K for large models (+~17 pts for PaLM-540B with 40 samples). Open question for me: are small-model errors too correlated for voting to help at k=5? → `experiments/gsm8k-selfconsistency/` (queued).
- **Program-aided LM / Program of Thoughts** (Gao et al. 2022 arXiv:2211.10435; Chen et al. 2022 arXiv:2211.12588): model writes Python, interpreter computes the answer. Removes arithmetic slips. Next experiment candidate.
- **Verifier / self-check**: model re-reads its answer, or a reward model ranks samples. Small models are weak self-critics; prefer external checks (execution, voting).

## Measured so far
- Qwen2.5-1.5B-Instruct Q4_K_M, greedy zero-shot CoT, GSM8K 100-q sample: **68%** (Qwen reports 73.2% full-precision); only 11% of outputs follow the `####` format. See `experiments/gsm8k-baseline/README.md`.
