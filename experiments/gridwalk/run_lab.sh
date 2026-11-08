#!/bin/bash
# gridwalk lab job: n=200 (indices 0-199), chunks of 25, resumable. See README Prediction.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
for OFF in $(seq 0 25 175); do
  python run.py --mode cot --n 25 --offset $OFF --steps 6 --out cot_greedy.jsonl | tail -1
  python run.py --mode cot --n 25 --offset $OFF --steps 6 --k 5 --temp 0.7 --sample_seed 81 --out cot_s.jsonl | tail -1
  python run.py --mode pal --n 25 --offset $OFF --steps 6 --k 5 --temp 0.7 --sample_seed 82 --out pal_s.jsonl | tail -1
  python analyze.py . > analysis.txt
done
cat analysis.txt
echo DONE
