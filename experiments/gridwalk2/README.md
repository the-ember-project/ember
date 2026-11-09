# gridwalk2 — CoT+program voting on generative state tracking, with a baseline that actually reasons (M2)

Follow-up to `experiments/gridwalk/`. There, H1 (MIX10 − greedy +8.0, p=0.017) passed, but greedy CoT skipped reasoning
on 98/200 questions and nearly all of the gain came from those (+14.3 there vs +2.0, p=0.84, where greedy reasoned).
This re-test fixes the baseline so the comparison is against a model that reasons.

## Changes vs gridwalk (decided from exploratory pilots, session #12, on indices 5300–5411 with seed 777; not reused)
- **One-shot chat demonstration for both formats** (same 5-step example task: CoT gets a worked trace, PAL gets a
  heading-index simulation program), given as a user/assistant turn before the real question. Pilots:
  user-message hints alone → CoT reasons but 0/12 (drops an instruction, wrong turn semantics); example pasted into
  the user message → T=0.7 samples parrot the example answer; chat-turn one-shot → no parroting.
- **10 steps** (one-shot makes 6 steps too easy: 5/6). 10-step pilot (n=12): greedy CoT 4/12, greedy PAL 5/12, with
  different questions right (errors look decorrelated).
- **Fresh tasks:** task seed 778 (seed 777's 10-step tasks share their first draws with gridwalk's 6-step tasks),
  test indices 0–199. Sample seeds 91 (CoT), 92 (PAL).
- Qwen2.5-1.5B-Instruct Q4_K_M, llama-cpp, CPU lab. Per question: greedy CoT, 5 CoT samples T=0.7, 5 PAL samples T=0.7.
  `run_lab.sh` chunks of 25, resumable; needs ~2 lab runs (~115 s/q).
- `analyze.py`: as gridwalk (plurality over parsed (x,y), abstentions dropped, ties → earliest listed sample, exact
  two-sided sign test on paired discordant questions, tokens relative to greedy) + greedy skip rate.

## Prediction (written before any test-index run)
Verdicts only at n=200 (resume if the first lab run stops early; no interim verdicts).
- Baseline: greedy CoT predicted 35% (range 20–50). **Pre-registered validity check:** greedy skip rate (<30 tokens)
  must be ≤20%, else the baseline is flagged degenerate and H1/H3 are not counted as M2 evidence.
- **H1 (primary):** MIX10 = vote(5 CoT + 5 PAL) − greedy CoT **≥ +3 points AND p < 0.05.** Predicted +10 (range 0 to
  +18); ~75% chance.
- **H2 (equal count, "mixing beats sampling"):** vote(3 CoT + 2 PAL) − vote(5 CoT) **≥ +2 points AND p < 0.05.**
  Predicted +4 (range −3 to +10); ~35% chance.
- **H3:** G3 = vote(greedy CoT + 3 PAL), greedy wins ties, − greedy **≥ +3 AND p < 0.05.** Predicted +5; ~50%.
- Reported, not pass/fail: single samples, vote(5P), vote(2C+3P), pass@11, PAL abstain rate, cost.
