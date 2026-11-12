#!/bin/bash
# CPU lab: G3 on MATH numeric test with Qwen2.5-3B Q4, positions 2250-2449, resumable (re-queue until n=200).
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy
timeout 7500 python ../math-g3/math_g3.py --repo Qwen/Qwen2.5-3B-Instruct-GGUF --file qwen2.5-3b-instruct-q4_k_m.gguf \
    --n 200 --offset 2250 --seed 9191 --out rows.jsonl | tail -3
python ../math-g3/analyze.py rows.jsonl > analysis.txt; cat analysis.txt
