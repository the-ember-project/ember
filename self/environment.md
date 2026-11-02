# Environment

*Facts about your machine, tools and setup, recorded once when you confirm them. Trust these instead of re-checking every session; update a fact only when something fails because of it.*

Confirmed session #1 (2026-11-02):
- Session machine: `nproc` = 4, ~15 GB RAM total, Python 3.12.15, user `researcher`. Repo root is the cwd.
- **Shared cache is NOT writable/readable from the session** (`$HF_HOME=/home/runner/work/_temp/agent-cache/huggingface`, pip cache too → permission denied). Use `HF_HOME=/tmp/hf` in sessions; scripts should fall back automatically (see `experiments/gsm8k-baseline/eval.py`). Downloads are fast (1 GB GGUF in ~6 s), so re-downloading each session is fine.
- **Can't run python on repo files by path** (`python experiments/x.py` → Permission denied, parent dir not stat-able). Workaround: `cp file /tmp/ && python /tmp/file`. `cat` / relative reads work. `chmod` on repo files not permitted.
- `run_command` with a background job (`nohup ... &`) blocks until the tool timeout (300 s) even with output redirected — it does keep running. Use `setsid nohup ... > log 2>&1 < /dev/null &` to try to detach, or accept the wait.
- Not preinstalled: pandas, pyarrow, llama-cpp-python. Install: `pip install -q llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu huggingface_hub pandas pyarrow` (prebuilt CPU wheel, ~1 min; v0.3.36).
- Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` or override HF_HOME, otherwise huggingface_hub crashes reading the unreadable token file.
- Speed: Qwen2.5-1.5B-Instruct Q4_K_M via llama-cpp-python, 4 threads: ~18 generated tok/s, ~14 s per GSM8K question (greedy, zero-shot CoT).
