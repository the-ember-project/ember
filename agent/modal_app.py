"""The Modal side of Ember's GPU jobs: the container image and the one function that runs a job.

agent/modal_jobs.py (run by the "Modal GPU jobs" workflow) calls run_job with the packed experiment
folder. Inside Modal's container, run_job unpacks it, runs the script with a time limit, and returns
what the job wrote or changed, packed the same way. Nothing else of the repository is sent, and the
Modal token stays in the workflow: it isn't passed to the container.

This file must import nothing from the repository at the top level: Modal loads it on its own inside
the container. The job logic is in execute(), plain Python, so it can be tested without Modal.
"""

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import time

import modal

GPU = "L4"                                  # 24 GB, supports bf16; a single GPU per job
CPU, MEMORY_MIB = 4.0, 24 * 1024            # like Kaggle's 4 cores; billed as max(request, use)
# Modal's list prices for functions (modal.com/pricing, checked 2026-11-28), in dollars per second,
# for what run_job asks for: about $1.18 an hour. agent/modal_jobs.py uses this to keep a month's
# use under its cap. Modal's own usage page is the real figure.
PRICE_PER_SECOND = 0.000222 + CPU * 0.0000131 + (MEMORY_MIB / 1024) * 0.00000222
MAX_JOB_MINUTES = 300                       # the workflow waits at most ~330 minutes per run
RETURN_FILE_LIMIT = 5 * 1024 * 1024         # the repository keeps nothing bigger (see save.sh)

# The image is built once and reused until this definition changes. The packages are what Ember's
# GPU jobs have used on Kaggle (vLLM, PEFT, TRL, Transformers); a job can still pip install more.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "peft", "trl", "transformers", "accelerate", "datasets", "huggingface_hub",
                 "numpy", "scipy", "pandas", "sentencepiece")
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App("ember-gpu", image=image)


def execute(bundle: bytes, entry: str, limit: int, work: str) -> bytes:
    """Unpack the job into work, run entry for at most limit minutes, return a tar.gz of the
    files it wrote or changed (plus gpu-output.txt with its printed output)."""
    os.makedirs(work, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bundle)) as t:
        names = [m.name for m in t.getmembers() if m.isfile()]
        t.extractall(work, filter="data")

    def digest(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    original = {n: digest(os.path.join(work, n)) for n in names}
    log_path = os.path.join(work, "gpu-output.txt")
    with open(log_path, "w") as log:
        log.write(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
                                 shell=True, capture_output=True, text=True).stdout)
        log.flush()
        argv = ([sys.executable] if entry.endswith(".py") else ["bash"]) + [entry]
        start = time.time()
        try:
            r = subprocess.run(argv, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=limit * 60)
            status = "exit code %d" % r.returncode
        except subprocess.TimeoutExpired:
            status = "stopped at its %d-minute limit" % limit
        log.write("\n[%s after %.0f min]\n" % (status, (time.time() - start) / 60))

    out, skipped = io.BytesIO(), []
    with tarfile.open(fileobj=out, mode="w:gz") as t:
        for dirpath, dirnames, filenames in os.walk(work, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
                                 and d != "__pycache__")
            for name in sorted(filenames):
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, work)
                if os.path.islink(p) or not os.path.isfile(p):
                    continue
                if rel == "gpu-output.txt" or (rel in original and digest(p) == original[rel]):
                    continue  # the log is added last; unchanged inputs aren't sent back
                size = os.path.getsize(p)
                if size > RETURN_FILE_LIMIT:
                    skipped.append(f"{rel} ({size / 1e6:.0f} MB)")
                    continue
                t.add(p, arcname=rel, recursive=False)
        if skipped:
            with open(log_path, "a") as log:
                log.write("[not returned, over 5 MB: " + ", ".join(skipped) + "]\n")
        t.add(log_path, arcname="gpu-output.txt", recursive=False)
    return out.getvalue()


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MIB, timeout=(MAX_JOB_MINUTES + 20) * 60, retries=0)
def run_job(bundle: bytes, entry: str, limit: int) -> bytes:
    return execute(bundle, entry, min(limit, MAX_JOB_MINUTES), "/work")
