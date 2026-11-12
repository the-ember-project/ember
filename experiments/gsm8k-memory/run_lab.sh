#!/bin/bash
# CPU lab: MoT-style memory at 1.5B. Phase 1 bank (400 train q), phase 2 eval (300 test q, 3 arms). Resumable; re-queue until n=300.
cd "$(dirname "$0")"
pip install -q --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.36 huggingface_hub pandas pyarrow scikit-learn
python - <<'P'
import json  # drop a partial last line left by a timeout kill
for fn in ("bank.jsonl", "rows.jsonl"):
    try: L = open(fn).read().splitlines()
    except FileNotFoundError: continue
    ok = []
    for l in L:
        try: json.loads(l); ok.append(l)
        except Exception: pass
    open(fn, "w").write("".join(x + "\n" for x in ok))
P
timeout 9600 python mem.py --n_bank 400 --n 300 --offset 300 | tail -5
[ -f rows.jsonl ] && python analyze.py rows.jsonl bank.jsonl > analysis.txt && cat analysis.txt
