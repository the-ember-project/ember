# m3-grpo-unsloth — does Unsloth make GRPO on a T4 cheaper? (issue #8)

Exploratory: speed/parity pilot for the training stack, not a capability claim. It decides whether future GRPO/DPO jobs use Unsloth.

## Setup
`pilot_u.py` (helpers copied from `m3-grpo/pilot.py`). One Kaggle job, two T4s in parallel, same environment:
- GPU0: Unsloth `FastLanguageModel` (fp16, LoRA r=16 on all proj, `fast_inference=True` vLLM rollouts, gpu_memory_utilization 0.35, Unsloth gradient checkpointing) + TRL GRPOTrainer.
- GPU1: stock TRL GRPOTrainer + peft + vLLM colocate (the m3-grpo / m3-grpo-main stack).
- Identical config: Qwen2.5-1.5B-Instruct, MATH-train prompts (same shuffle seed 31), 4 prompts × 8 completions per optimizer step
  (bs 8, ga 4), max 1024 new tokens, T=1.0, beta 0, lr 1e-5, GOLD reward, 12 steps, seed 7.
- If the Unsloth+vLLM run fails, a 3-step fallback runs Unsloth with HF generation.
- CPU dry run (0.5B, 1 step): stock path OK; Unsloth path cannot be tested on CPU.

## Decision rule (written before the run)
Switch later GRPO jobs to Unsloth only if all hold: (a) the Unsloth run finishes 12 steps without error; (b) its median
s/step (steps 2–12) ≤ 0.67× the stock run's in the same job; (c) parity sanity: mean `sampling/sampling_logp_difference`
≤ 0.01 and mean GOLD reward over 12 steps within ±0.15 of stock's (same prompts). Install time is reported separately and
counts against the gain (amortised over a ~100-step run). Never mix Unsloth and stock arms in one comparison.
Reference: m3-grpo pilot stock = 69 s/step (median), 12.1 GB.

## Results
(pending; queued on Kaggle `pilot_u.py 60` after m3-grpo-main)

## Results (session #37, exploratory; Kaggle 19 GPU min)
- **Unsloth + fast_inference vLLM: worked.** 12 steps, load 141 s, median **57.5 s/step** (steps 2+; range 39–97).
  Skips zero-advantage rows (25–75% per step). No OOM at bs 8 × ga 4 with vLLM at 0.35.
- **Stock TRL+peft+vLLM (GPU1): CUDA OOM in step 1** (14.3 GB used, needed +340 MB). Same failure hit all three arms
  of m3-grpo-main run 2 at step 3. So no same-job s/step ratio exists.
- Decision rule as written: against the historic stock 69 s/step (m3-grpo pilot run 2) the ratio is 0.83×, above the
  0.67× bar; the logp-parity check can't be computed (Unsloth logs no sampling_logp_difference). **Verdict: don't switch.**
  Fix stock instead: bs 2 × ga 16 (same 32 completions/step; logits memory /4) + OOM retry at bs 1 (m3-grpo-main run 3).
- Noted advantage: Unsloth fit the bs-8 config where stock did not — revisit if stock at bs 2 proves too slow.
