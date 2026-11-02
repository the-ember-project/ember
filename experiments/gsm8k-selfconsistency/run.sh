#!/bin/bash
# Lab job: self-consistency k=5 on the 100-question baseline sample, plus greedy on 200 more questions.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
EVAL=../gsm8k-baseline/eval.py
# 1) self-consistency on the same 100 questions as the greedy baseline (resumable)
python $EVAL --n 100 --offset 0 --k 5 --temp 0.7 --out results_sc_k5_t0.7_n100.jsonl | tail -3
# 2) greedy on 200 more questions (offset 100) for a tighter baseline
python $EVAL --n 200 --offset 100 --out ../gsm8k-baseline/results_qwen1.5b_greedy_off100_n200.jsonl | tail -3
echo DONE
