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
- Report "met threshold but p large" as such, never as "X helps".
