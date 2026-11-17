#!/bin/bash
# CPU lab: math-sc7, resumable. ~45 s/q expected on the lab machine.
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy
python - <<'P'
import json
try: L = open("cs2.jsonl").read().splitlines()
except FileNotFoundError: L = []
ok = []
for l in L:
    try: json.loads(l); ok.append(l)
    except Exception: pass
open("cs2.jsonl", "w").write("".join(x + "\n" for x in ok))
P
timeout ${T:-16800} python sc7.py --n 250 --out cs2.jsonl | tail -3
python analyze.py > analysis.txt; cat analysis.txt
