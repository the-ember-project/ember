#!/bin/bash
# CPU lab: third answer format on MATH (1.5B). Resumable; re-queue until n=250.
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
timeout 12600 python f3.py --n 250 --out rows.jsonl | tail -3
python analyze.py rows.jsonl > analysis.txt; cat analysis.txt
