#!/bin/bash
# Lab job (resumable): 4 arms on fresh GSM8K offsets 300-599, in two chunks so partial runs stay paired.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
E=../gsm8k-baseline/eval.py; P=../gsm8k-pal/pal.py
for OFF in 300 450; do
  python $P --n 150 --offset $OFF --out pal_greedy_off300_n300.jsonl | tail -1
  python $E --n 150 --offset $OFF --out cot_greedy_off300_n300.jsonl | tail -1
  python $E --n 150 --offset $OFF --k 5 --temp 0.7 --sample_seed 777 --out cot_k5_off300_n300.jsonl | tail -1
  python $P --n 150 --offset $OFF --k 5 --temp 0.7 --sample_seed 777 --out pal_k5_off300_n300.jsonl | tail -1
done
python analyze.py > analysis.txt; cat analysis.txt
echo DONE
