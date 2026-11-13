#!/bin/bash
# Lab job (resumable): 1.5B PAL k=3 samples on GSM8K offsets 0-299, then compare with stored 1.5B and 3B greedy.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
P=../gsm8k-pal/pal.py
for OFF in 0 100 200; do
  python $P --n 100 --offset $OFF --k 3 --temp 0.7 --sample_seed 8484 --out pal_1.5b_k3.jsonl | tail -1
  python compare.py > compare.txt || true
done
cat compare.txt
echo DONE
