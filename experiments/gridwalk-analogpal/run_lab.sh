#!/bin/bash
# gridwalk-analogpal lab job: analog-demo PAL, 5 samples T=0.7, same 200 tasks as gridwalk2 (task seed 778). Resumable.
set -e
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow
for OFF in $(seq 0 25 175); do
  python run.py --mode pal --n 25 --offset $OFF --steps 10 --task_seed 778 --shot 1 --k 5 --temp 0.7 --sample_seed 92 --out pal_analog.jsonl | tail -1
  python analyze.py > analysis.txt || true
done
cat analysis.txt
echo DONE
