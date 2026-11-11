#!/bin/bash
# CPU lab: G3 on MATH numeric test, resumable (re-queue until analysis shows n=250).
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy
timeout 17100 python math_g3.py --n 250 --offset 2000 --out rows.jsonl | tail -3
python analyze.py rows.jsonl > analysis.txt; cat analysis.txt
