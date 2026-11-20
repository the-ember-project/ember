#!/bin/bash
# math-conf lab run (resumable). Needs experiments/math-verify/gpu-output/out/gen_1.5B.jsonl.gz.
set -e
cd "$(dirname "$0")"
pip install -q llama-cpp-python==0.3.36 --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu huggingface_hub numpy
python conf.py --out conf_1.5B.jsonl
python analyze.py > analysis.txt 2>&1 || true
cat analysis.txt
