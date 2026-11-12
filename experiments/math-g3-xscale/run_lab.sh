#!/bin/bash
# CPU lab: 1.5B G3 on the 3B experiment's MATH questions (positions 2250-2449). Resumable; re-queue until n=200.
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy
python - <<'P'
import json
try: L = open("rows.jsonl").read().splitlines()
except FileNotFoundError: L = []
ok = []
for l in L:
    try: json.loads(l); ok.append(l)
    except Exception: pass
open("rows.jsonl", "w").write("".join(x + "\n" for x in ok))
P
timeout 8400 python ../math-g3/math_g3.py --n 200 --offset 2250 --seed 9393 --out rows.jsonl | tail -3
python ../math-g3/analyze.py rows.jsonl > analysis.txt; cat analysis.txt
python compare.py > compare.txt; cat compare.txt
