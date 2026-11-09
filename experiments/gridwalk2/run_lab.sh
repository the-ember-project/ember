#!/bin/bash
# gridwalk2 lab job: 10-step tasks, task seed 778, indices 0-199, one-shot chat demo for both formats.
# Chunks of 25, resumable (needs ~2 lab runs). See README Prediction.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
A="--steps 10 --task_seed 778 --shot 1"
for OFF in $(seq 0 25 175); do
  python run.py --mode cot --n 25 --offset $OFF $A --out cot_greedy.jsonl | tail -1
  python run.py --mode cot --n 25 --offset $OFF $A --k 5 --temp 0.7 --sample_seed 91 --out cot_s.jsonl | tail -1
  python run.py --mode pal --n 25 --offset $OFF $A --k 5 --temp 0.7 --sample_seed 92 --out pal_s.jsonl | tail -1
  python analyze.py . > analysis.txt
done
cat analysis.txt
echo DONE
