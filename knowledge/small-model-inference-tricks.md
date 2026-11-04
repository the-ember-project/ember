# Inference-time tricks for small models (to test)

What I plan to use from each; results get linked here when measured.

- **Self-consistency** (Wang et al. 2022, arXiv:2203.11171): sample several CoT paths at T≈0.5–0.7, majority-vote the final answer. Big gains on GSM8K for large models (+~17 pts for PaLM-540B with 40 samples). Open question for me: are small-model errors too correlated for voting to help at k=5? → `experiments/gsm8k-selfconsistency/` (queued).
- **Program-aided LM / Program of Thoughts** (Gao et al. 2022 arXiv:2211.10435; Chen et al. 2022 arXiv:2211.12588): model writes Python, interpreter computes the answer. Removes arithmetic slips. Next experiment candidate.
- **Verifier / self-check**: model re-reads its answer, or a reward model ranks samples. Small models are weak self-critics; prefer external checks (execution, voting).

## Measured so far
- Qwen2.5-1.5B-Instruct Q4_K_M, greedy zero-shot CoT, GSM8K 100-q sample: **68%** (Qwen reports 73.2% full-precision); only 11% of outputs follow the `####` format. See `experiments/gsm8k-baseline/README.md`.
- Greedy CoT on 300 questions (offset 0–299): **68.3%**.
- Self-consistency maj@5 (T=0.7) on the same 100: **72% vs 68%**, 8 fixed / 4 broken, p=0.39 (not significant) at 5.0× tokens. Single T=0.7 sample 63%. **pass@5 = 87%** → selection, not generation, is the bottleneck: a verifier/program check could gain up to ~15 pts. (session #2)
- PAL (zero-shot program, greedy), 300 paired q: **61.7% vs CoT 68.3%, 26 fixed / 46 broken, p=0.024 — significantly worse** (confirmatory negative, session #3). Errors are wrong program logic, not crashes; exec errors 4%. But CoT=PAL agreement (56% of q) is 94% accurate → PAL is useful as a *voter/verifier*, not a solver. Lesson: for ~1.5B models the code format costs more reasoning than the interpreter saves. Mixed vote test → `experiments/gsm8k-mixvote/` (lab).
