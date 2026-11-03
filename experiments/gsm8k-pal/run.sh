#!/bin/bash
# Lab job: PAL on 200 fresh questions (offset 100); also finishes the offset-0 run if incomplete (resumable).
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
python pal.py --n 100 --offset 0 --out results_pal_n100.jsonl | tail -2
python pal.py --n 200 --offset 100 --out results_pal_off100_n200.jsonl | tail -2
echo DONE
