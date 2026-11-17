#!/bin/bash
# CPU lab: math-3b-sc, resumable (re-queue until analyze shows greedy n=250 and sampled n=250).
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow sympy
python - <<'P'
import json
for f in ("g.jsonl", "s.jsonl"):
    try: L = open(f).read().splitlines()
    except FileNotFoundError: continue
    ok = []
    for l in L:
        try: json.loads(l); ok.append(l)
        except Exception: pass
    open(f, "w").write("".join(x + "\n" for x in ok))
P
timeout ${T:-16800} python run.py | tail -3
python analyze.py > analysis.txt; cat analysis.txt
