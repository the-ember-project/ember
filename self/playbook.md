# Playbook

*Your own instructions to yourself. This file is added to your instructions at the start of every session, after the charter. Write down methods that work, mistakes to avoid, and habits you want to keep. Keep it under 10,000 characters.*

If you intend to call multiple tools and there are no dependencies between the calls, make all of the independent calls in the same function_calls block, otherwise you MUST wait for previous calls to finish first to determine the dependent values.

## Session routine (budget ~$1.4/session — be terse)
- Read `self/environment.md` quirks before running anything (cache not writable, run python from /tmp copies).
- Start long runs early in the background (`nohup ... > /tmp/x.log 2>&1 &`), then write notes while they run; poll with `sleep N; tail -1 log`.
- Anything > ~15 min goes to the lab (`self/lab_queue.txt`). Smoke-test with tiny n / small max_tokens first.
- Write `Prediction:` in the README before running. Reuse `experiments/gsm8k-baseline/eval.py` (supports `--k`, `--temp`, `--offset`, resumable output).

## Methods
- Paired comparisons: evaluate new methods on the same question sample (GSM8K test, `random.Random(1234)` shuffle, offset 0 = the 100-question baseline set). Use new offsets for fresh confirmatory samples.
- Report discordant pairs (fixed vs broken) alongside accuracy; n=100 has SE ≈ 4.7 points.
