#!/bin/bash
# Lab job 2: (a) 0.5B G3 re-run with a CoT-eliciting system prompt on fresh offsets 400-799; (b) resume 3B to n=300.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
E=../gsm8k-baseline/eval.py; P=../gsm8k-pal/pal.py
M05="--repo Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf"
M3="--repo Qwen/Qwen2.5-3B-Instruct-GGUF --file qwen2.5-3b-instruct-q4_k_m.gguf"
SYS="You are a helpful math tutor. Solve the problem step by step, writing out each calculation. Only at the very end, write the final answer on its own line as #### <number>."
for OFF in $(seq 400 100 700); do
  python $E $M05 --n 100 --offset $OFF --system "$SYS" --out cot_0.5b_v2.jsonl | tail -1
  python $P $M05 --n 100 --offset $OFF --k 3 --temp 0.7 --sample_seed 7171 --out pal_0.5b_v2.jsonl | tail -1
  python analyze.py 0.5b_v2 > analysis_0.5b_v2.txt
done
cat analysis_0.5b_v2.txt
for OFF in $(seq 0 50 250); do
  python $E $M3 --n 50 --offset $OFF --out cot_3b.jsonl | tail -1
  python $P $M3 --n 50 --offset $OFF --k 3 --temp 0.7 --sample_seed 6161 --out pal_3b.jsonl | tail -1
  python analyze.py 3b > analysis_3b.txt
done
cat analysis_3b.txt
echo DONE
