#!/bin/bash
# Lab job (resumable): confirm greedy-anchored cheap mix G3 on GSM8K test offsets 900-1318 (n=419), Q4 llama-cpp.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
E=../gsm8k-baseline/eval.py; P=../gsm8k-pal/pal.py
for OFF in 900 1110; do
  N=$(( OFF == 900 ? 210 : 209 ))
  python $E --n $N --offset $OFF --out cot_greedy_off900.jsonl | tail -1
  python $P --n $N --offset $OFF --k 4 --temp 0.7 --sample_seed 5151 --out pal_k4_off900.jsonl | tail -1
done
python analyze.py > analysis.txt; cat analysis.txt
echo DONE
