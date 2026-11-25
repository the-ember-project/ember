#!/bin/bash
# Lab job (resumable): cap-fix re-check for gsm8k-orm (see README "Cap-fix re-check").
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q transformers accelerate
python uncap.py --src cot3b_greedy.jsonl --dst cot3b_greedy_u.jsonl --repo Qwen/Qwen2.5-3B-Instruct-GGUF \
    --file qwen2.5-3b-instruct-q4_k_m.gguf --temp 0 | tail -1
python uncap.py --src ../gsm8k-g3confirm/cot_greedy_off900.jsonl --dst greedy15_u.jsonl --temp 0 --limit 210 | tail -1
python uncap.py --src cot_k7.jsonl --dst cot_k7_u.jsonl --temp 0.7 --seed 7171 | tail -1
python rescore.py | tail -1
python analyze.py --suffix _u > analysis_u.txt; cat analysis_u.txt
echo DONE
