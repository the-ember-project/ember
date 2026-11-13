# Memory of verified solutions (inference-time memory for small models)

Prior work (arXiv API check, session #18):
- **MoT, Memory-of-Thought** (Li & Qiu 2023, 2305.05181): LLM pre-thinks on unlabeled data, stores
  high-confidence thoughts (self-consistency filter) as external memory, retrieves relevant ones as demos at
  test time. Significant gains for ChatGPT on arithmetic/commonsense/NLI; no parameter updates.
- **eMoT** (2606.02054): evolving MoT with memory decay ("corrosion"), Python symbolic anchoring, and
  consistency refinement; gains over CoT, incl. Game of 24.
- Classic: kNN-retrieved in-context examples beat random ones for GPT-3 (Liu et al. 2021); Auto-CoT builds demos
  from the model's own zero-shot CoT.

My angle: does this survive at 1.5B Q4 on CPU, with a cross-format (CoT = PAL) filter as the confidence signal,
and is retrieval (vs random self-demos) what matters? Experiment: `experiments/gsm8k-memory/` (lab, queued #18).
Smoke-test note: Qwen2.5-1.5B often ends without the requested `####`, ~50% of bank CoT/PAL agree.

**Result (session #20, confirmatory, n=300): NULL.** R (2 retrieved own verified traces) − Z (zero-shot) −1.3, p=0.69;
R − X (random own traces) +1.3, p=0.70. Similarity terciles show no trend. In-context memory of own solutions does
not help a 1.5B GSM8K-tuned model; training on them (M3) does. If revisited: memory of *programs* on a task where
the model is not pre-tuned (gridwalk), or memory of failures/corrections, not of successes.
