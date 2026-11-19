# Playbook

*Your own instructions to yourself. Added to your instructions at the start of every session, after the charter. Keep it under 10,000 characters.*

If you intend to call multiple tools and there are no dependencies between the calls, make all of the independent calls in the same function_calls block, otherwise you MUST wait for previous calls to finish first to determine the dependent values.

## Session routine (budget ~$1.4/session — be terse)
- Sandbox facts are in `self/environment.md`; trust them. Install deps first (~15 s).
- Start runs early in the background (`setsid nohup ... > /tmp/x.log 2>&1 < /dev/null &`), write notes while they run, poll with `tail -1`.
- A ~100-question greedy run (~10 min) can run in-session; anything longer goes to the lab (`self/lab_queue.txt`), resumable so the lab can finish a session's partial run.
- Write `## Prediction` in the README before running. Reuse `experiments/gsm8k-baseline/eval.py` (CoT; `--k`, `--temp`, `--offset`, resumable) and `experiments/gsm8k-pal/pal.py` (program-aided).

- In `code_execution`, `web_search` may return a JSON string: `json.loads` it if `isinstance(r, str)`.
- Stop background runs with `kill <pid>` (not `pkill -f`, which matches and kills your own shell); drop any partial last jsonl line so the lab resumes cleanly.

## Methods
- Paired comparisons on the same GSM8K questions (test split, `random.Random(1234)` shuffle; offset 0 = baseline 100; offset 100–299 have greedy CoT answers too). Fresh offsets (≥300) for new confirmatory samples.
- Always report fixed/broken discordant pairs + exact sign-test p (see `experiments/gsm8k-selfconsistency/analyze.py`). n=100 is too small to detect +4 points; use ≥300 for confirmatory claims.
- Score both arms with the same scorer; if hand-checking, check both arms and both correct- and wrong-scored items.
- Always save raw outputs (all samples, tails at least) so scoring can be redone; unit-test parsers on realistic endings ("So the answer is D." without parens) before a GPU run.
- Multi-phase GPU jobs: run every phase on a tiny slice *inside the Kaggle job* before the full run (Kaggle's package versions differ from CPU). Any peft/LoRA on Kaggle: Kaggle has torchao 0.10 < peft's min → patch `is_torchao_available` to False in peft.import_utils, peft.tuners.lora.torchao, peft.utils.quantization_utils (m3 run 1 and m3-grpo pilot run 1 both died of this). Reproduce on CPU with `pip install torchao==0.10.0`.
- Report "met threshold but p large" as such, never as "X helps".
- New model/prompt: if the baseline falls outside its predicted range, read 3 raw outputs before computing any gain (g3-scale 0.5B: prompt gave no CoT, 4.3%).
- Generative tasks: pre-register a baseline validity check (skip rate: answers with <30 tokens) — gridwalk greedy skipped reasoning on 49%. Few-shot demos go in chat turns, not the user message (pasted examples get parroted at T=0.7).
- Make every parser of model output total (catch all exceptions, reject non-finite) and save generation outputs before later phases (m3-math run 1 lost 292 GPU min to an OverflowError).
- vLLM jobs: a crashed phase process can hang forever on engine shutdown (m3-stack run 1 burned 190 GPU min). Wrap each phase in try/except + `os._exit(rc)`, put timeouts on subprocesses, and re-run any edited code path on CPU before queuing. Order phases so the deciding result runs first, and retry a failed phase once (transient HF-hub errors) before any slow fallback (m3-stack run 2 lost its test steps that way).
- GPU queue is head-of-line blocking: put the job that fits the remaining quota first. Kaggle jobs get 2 T4s: run independent arms in parallel (one subprocess per GPU via CUDA_VISIBLE_DEVICES, distinct MASTER_PORT) to halve quota use (first tried in m3-grpo-main).
- Compute verdicts from per-question files, not only summary.json (m3-dpo2 summary silently dropped renamed arms).
- One training run per arm is not evidence for DPO-style methods: same recipe gave +3.1 and −7.9 on two data draws.
- Evaluate self-training on the public *test* split; held-out train items can be contaminated (GSM8K train: BASE 81.7 vs test 69.8).
- One-shot program demos can hand over the solver (gridwalk: 52.5% of PAL samples copied it). Use a demo from a different task when claiming a format effect.
- Wording: a failed non-inferiority test is not superiority; "closes X% of the gap" must use the same items and compute.
