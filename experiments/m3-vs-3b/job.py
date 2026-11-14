"""Kaggle job: 3B greedy eval on m3-dpo3's held-out GSM8K-train items + BBH date, via the identical run.py eval phase."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("M3B_MODEL", "Qwen/Qwen2.5-3B-Instruct")
extra = os.environ.get("M3B_ARGS", "").split()
backend = "hf"
try:
    if os.environ.get("M3B_SKIP_VLLM"): raise RuntimeError("skip (smoke)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "peft"], timeout=1200, check=True)
    subprocess.run([sys.executable, "-c", "from vllm import LLM, SamplingParams; "
                    f"l=LLM('{MODEL}', dtype='half', max_model_len=1024, gpu_memory_utilization=0.5); "
                    "print('VLLM_OK', l.generate(['2+2='], SamplingParams(max_tokens=8))[0].outputs[0].text)"],
                   timeout=900, check=True)
    backend = "vllm"
except Exception as e:
    print("vLLM unavailable, falling back to HF:", e, flush=True)
print("backend", backend, flush=True)
os.makedirs("/tmp/w3b", exist_ok=True)
subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--phase", "eval", "--backend", backend,
                "--model", MODEL, "--work", "/tmp/w3b", "--out", HERE, "--bs", "32"] + extra, check=True)
