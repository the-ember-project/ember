#!/bin/bash
# CPU lab: math-selfverify on stored tails, resumable (~30 s/q on the session machine, 250 q).
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy numpy
python - <<'P'
import json, os
L = open("sv_tails.jsonl").read().splitlines() if os.path.exists("sv_tails.jsonl") else []
ok = []
for l in L:
    try: json.loads(l); ok.append(l)
    except Exception: pass
open("sv_tails.jsonl", "w").write("".join(x + "\n" for x in ok))
P
timeout ${T:-12000} python sv.py --n 250 --out sv_tails.jsonl 2>&1 | grep -v Warning | tail -3
python analyze.py sv_tails.jsonl > analysis.txt; cat analysis.txt
