#!/bin/bash
# gridwalk2-3b: Qwen2.5-3B Q4, 10-step tasks, task seed 778, FRESH indices 200-349, one-shot. Chunks of 25, resumable.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
A="--steps 10 --task_seed 778 --shot 1 --repo Qwen/Qwen2.5-3B-Instruct-GGUF --file qwen2.5-3b-instruct-q4_k_m.gguf"
for OFF in $(seq 200 25 325); do
  python run.py --mode cot --n 25 --offset $OFF $A --out cot_greedy.jsonl | tail -1
  python run.py --mode pal --n 25 --offset $OFF $A --out pal_greedy.jsonl | tail -1
  python run.py --mode pal --n 25 --offset $OFF $A --k 3 --temp 0.7 --sample_seed 93 --out pal_s.jsonl | tail -1
  python analyze.py > analysis.txt
done
cat analysis.txt
echo DONE
