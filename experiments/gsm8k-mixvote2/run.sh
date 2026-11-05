#!/bin/bash
# Lab job (resumable): 3 arms on fresh GSM8K offsets 600-899, two chunks so partial runs stay paired.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
E=../gsm8k-baseline/eval.py; P=../gsm8k-pal/pal.py
for OFF in 600 750; do
  python $E --n 150 --offset $OFF --out cot_greedy_off600_n300.jsonl | tail -1
  python $P --n 150 --offset $OFF --k 5 --temp 0.7 --sample_seed 4242 --out pal_k5_off600_n300.jsonl | tail -1
  python $E --n 150 --offset $OFF --k 2 --temp 0.7 --sample_seed 4242 --out cot_k2_off600_n300.jsonl | tail -1
done
python analyze.py > analysis.txt; cat analysis.txt
echo DONE
