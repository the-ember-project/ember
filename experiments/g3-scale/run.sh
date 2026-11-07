#!/bin/bash
# Lab job (resumable): G3 = vote(greedy CoT + 3 PAL samples, greedy wins ties) at other model sizes. Q4_K_M llama-cpp.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
E=../gsm8k-baseline/eval.py; P=../gsm8k-pal/pal.py
N05=${N05:-400}; N3=${N3:-300}
M05="--repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf"
M3="--repo Qwen/Qwen2.5-3B-Instruct-GGUF --file qwen2.5-3b-instruct-q4_k_m.gguf"
python $E $M05 --n $N05 --offset 0 --out cot_0.5b.jsonl | tail -1
python $P $M05 --n $N05 --offset 0 --k 3 --temp 0.7 --sample_seed 6161 --out pal_0.5b.jsonl | tail -1
python analyze.py 0.5b > analysis_0.5b.txt; cat analysis_0.5b.txt
# 3B in chunks of 50 so a time-limit stop loses little (resumable)
for OFF in $(seq 0 50 $((N3-50))); do
  python $E $M3 --n 50 --offset $OFF --out cot_3b.jsonl | tail -1
  python $P $M3 --n 50 --offset $OFF --k 3 --temp 0.7 --sample_seed 6161 --out pal_3b.jsonl | tail -1
  python analyze.py 3b > analysis_3b.txt
done
cat analysis_3b.txt
echo DONE
