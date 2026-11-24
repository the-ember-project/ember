#!/bin/bash
# Lab job (resumable): 3B greedy + 1.5B 7 CoT samples on GSM8K test offsets 900-1109, then RM scoring on CPU.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q transformers accelerate
E=../gsm8k-baseline/eval.py
python $E --repo Qwen/Qwen2.5-3B-Instruct-GGUF --file qwen2.5-3b-instruct-q4_k_m.gguf \
    --n 210 --offset 900 --out cot3b_greedy.jsonl | tail -1
python $E --n 210 --offset 900 --k 7 --temp 0.7 --sample_seed 6262 --out cot_k7.jsonl | tail -1
python score.py | tail -2
python analyze.py > analysis.txt; cat analysis.txt
echo DONE
