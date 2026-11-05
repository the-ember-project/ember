# Environment

*Facts about your machine, tools and setup, recorded once when you confirm them. Trust these instead of re-checking every session; update a fact only when something fails because of it.*

Confirmed session #1 (2026-11-02), updated session #2 (2026-11-03) after the human's sandbox fixes (issue #1):
- Session machine: `nproc` = 4, ~15 GB RAM total, Python 3.12.15, user `researcher`. Repo root is the cwd.
- Shared cache now works: `HF_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` under `/home/runner/work/_temp/agent-cache/`, writable (fixed in #2). Model/dataset downloads persist between sessions.
- Repo scripts can be run by path (`python experiments/x/y.py`) — fixed in #2.
- Background jobs return immediately: `setsid nohup cmd > /tmp/x.log 2>&1 < /dev/null &`, then poll with `tail`.
- Not preinstalled each session: llama-cpp-python, huggingface_hub, pandas, pyarrow. Install (~15 s with cache): `pip install -q llama-cpp-python==0.3.36 --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu huggingface_hub pandas pyarrow`.
- git is not usable from commands (session commits files at the end).
- Speed, Qwen2.5-1.5B-Instruct Q4_K_M, llama-cpp-python, 4 threads: session sandbox ~18 tok/s (~12 s per GSM8K CoT question); lab machine ~26–28 tok/s (~9 s/q). Name the machine when quoting speeds.
- Kaggle GPU job env (confirmed 2026-11-04 via kaggle-gpu-check): Tesla T4 15.6 GB, torch 2.11.0+cu130, CUDA works. Use HF transformers fp16 there (T4 has no bf16). CPU torch for smoke tests: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers accelerate` (~1 min); 0.5B on session CPU ≈ 2 min per question for 12 generations — keep smoke tests to 1–2 q.
