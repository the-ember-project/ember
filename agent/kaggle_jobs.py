"""GPU jobs on Kaggle: runs the experiments the agent queued for a GPU, with no Claude calls.

The agent adds lines to self/gpu_queue.txt during a session:

    experiments/my-experiment/train.py 120

(a .py or .sh script under experiments/, then an optional time limit in minutes). After the
session, the "GPU jobs" workflow sends each one to Kaggle as a private script with a GPU, one at
a time, waits for it, and downloads what it wrote into gpu-output/ next to the script. Results
are logged in self/gpu_log.md and self/service_log.md, which the next session reads. This
workflow never edits the queue itself; it records the lines it took, and the session removes them.

The Kaggle key stays in this workflow. The agent's code runs only on Kaggle, never here, and
only the files of its own experiment folder are sent there (no links, nothing from outside it).
Usage is kept in state/kaggle_usage.json and capped per rolling 7 days (GPU_WEEKLY_MINUTES).
"""

import base64
from collections import Counter
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import EXPERIMENTS, ROOT, gpu_job_id, parse, provenance  # noqa: E402  (same queue rules as the lab)
import safefs  # noqa: E402  (symlink-safe reads and writes of the agent's files)

QUEUE = ROOT / "self" / "gpu_queue.txt"
LOG = ROOT / "self" / "gpu_log.md"
SERVICE_LOG = ROOT / "self" / "service_log.md"
USAGE = ROOT / "state" / "kaggle_usage.json"
# Queue lines this workflow took that the session has since removed from the queue. This
# workflow never edits self/gpu_queue.txt itself: it can run at the same time as a session in
# which the agent edits the queue, and both saving the file would collide. So it records each
# line it takes in USAGE, skips lines still waiting to be removed, and the session removes them.
QUEUE_DONE = ROOT / "state" / "gpu_queue_done.json"

USER = os.environ.get("KAGGLE_USERNAME") or "focarius"
WEEKLY_MINUTES = int(os.environ.get("GPU_WEEKLY_MINUTES") or 1440)   # 24 h, of Kaggle's ~30 h a week
POLL_MINUTES = int(os.environ.get("GPU_POLL_MINUTES") or 320)        # how long this run waits
MAX_JOB_MINUTES = 540                                                  # Kaggle stops sessions at 12 h
OVERHEAD = 10                                                          # start-up and download, per job
GIVE_UP_HOURS = 13                                                     # after this, a job counts as lost
# Ask for a T4 by name (Kaggle's "GPU T4 x2", its default GPU) rather than any GPU, so a job never
# lands on an older card that current PyTorch builds don't support (e.g. the retired Pascal P100).
ACCELERATOR = "NvidiaTeslaT4"
MAX_FILE, MAX_BUNDLE = 500_000, 2_000_000
SKIP = {"gpu-output", "__pycache__", ".git"}
DONE = ("complete", "error", "cancel")

WRAPPER = r'''# Ember GPU job, made by agent/kaggle_jobs.py: unpacks the experiment folder, runs it with a time limit.
import base64, hashlib, io, os, subprocess, sys, tarfile, time
BUNDLE = "{bundle}"
ENTRY, LIMIT = {entry!r}, {limit}
WORK = "/kaggle/working"
data = base64.b64decode(BUNDLE)
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    names = [m.name for m in t.getmembers() if m.isfile()]
    t.extractall(WORK, filter="data")
def digest(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
original = {{n: digest(os.path.join(WORK, n)) for n in names}}
log = open(os.path.join(WORK, "gpu-output.txt"), "w")
log.write(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
                         shell=True, capture_output=True, text=True).stdout)
log.flush()
argv = ([sys.executable] if ENTRY.endswith(".py") else ["bash"]) + [ENTRY]
start = time.time()
try:
    r = subprocess.run(argv, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, timeout=LIMIT * 60)
    status = "exit code %d" % r.returncode
except subprocess.TimeoutExpired:
    status = "stopped at its %d-minute limit" % LIMIT
log.write("\n[%s after %.0f min]\n" % (status, (time.time() - start) / 60))
log.close()
for n in names:  # send back only what the job wrote or changed
    p = os.path.join(WORK, n)
    if os.path.isfile(p) and digest(p) == original[n]:
        os.remove(p)
'''


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return f"{now():%Y-%m-%d %H:%M} UTC"


def load_usage() -> dict:
    try:
        return json.loads(USAGE.read_text())
    except Exception:
        return {"jobs": []}


def save_usage(u: dict) -> None:
    # Written to a temporary file and renamed, so a stopped job never leaves it half-written.
    USAGE.parent.mkdir(exist_ok=True)
    tmp = USAGE.with_name(USAGE.name + ".tmp")
    tmp.write_text(json.dumps(u, indent=2) + "\n")
    os.replace(tmp, USAGE)


def used_this_week(u: dict) -> int:
    since = now() - dt.timedelta(days=7)
    return sum(j["charged"] for j in u["jobs"] if dt.datetime.fromisoformat(j["submitted"]) > since)


def kaggle(*args: str, timeout: int = 600) -> tuple[int, str]:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    text = (r.stdout + r.stderr).strip()
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        text = text.replace(token, "[key]")
    return r.returncode, text[-600:]


def bundle(folder: Path) -> str:
    """The experiment folder's own small files, as a base64 tar.gz. No links, nothing outside it."""
    buf, total = io.BytesIO(), 0
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not (Path(dirpath) / d).is_symlink())
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.is_symlink() or not p.is_file() or name == "lab-output.txt":
                    continue
                if folder.resolve() not in p.resolve().parents:
                    continue
                size = p.stat().st_size
                if size > MAX_FILE:
                    continue
                total += size
                if total > MAX_BUNDLE:
                    raise ValueError(f"its folder's files add up to over {MAX_BUNDLE // 1_000_000} MB; "
                                     "download big data inside the job instead")
                t.add(p, arcname=str(p.relative_to(folder)), recursive=False)
    return base64.b64encode(buf.getvalue()).decode()


def slug_for(folder: Path) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-")
    return f"ember-{s}"[:50].rstrip("-")


def submit(script: Path, minutes: int) -> str:
    folder = script.parent
    slug = slug_for(folder)
    with tempfile.TemporaryDirectory() as tmp:
        code = WRAPPER.format(bundle=bundle(folder), entry=str(script.relative_to(folder)), limit=minutes)
        Path(tmp, "run.py").write_text(code)
        meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": ACCELERATOR,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
        Path(tmp, "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        rc, out = kaggle("kernels", "push", "-p", tmp)
    if rc != 0 or "error" in out.lower():
        raise RuntimeError(f"Kaggle refused it: {out}")
    return f"{USER}/{slug}"


def status(kernel: str) -> str:
    rc, out = kaggle("kernels", "status", kernel, timeout=120)
    m = re.search(r'status\s+"?(?:\w+\.)?(\w+)', out, re.IGNORECASE)
    return m.group(1).lower() if (rc == 0 and m) else "unknown"


def fetch(job: dict) -> str:
    dest = ROOT / job["folder"] / "gpu-output"
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    rc, out = kaggle("kernels", "output", job["kernel"], "-p", str(dest), timeout=900)
    for p in dest.rglob("*"):
        if p.is_symlink():
            p.unlink()
    return "" if rc == 0 else f" (download failed: {out[-200:]})"


def finish(job: dict, state: str, u: dict) -> None:
    elapsed = (now() - dt.datetime.fromisoformat(job["submitted"])).total_seconds() / 60
    job["charged"] = int(min(elapsed, job["limit"] + OVERHEAD)) + 1
    job["status"], job["pending"], job["finished"] = state, False, now().isoformat(timespec="seconds")
    note = fetch(job)
    save_usage(u)
    entry = (f"- {stamp()} `{job['line']}` on Kaggle ({job['kernel']}): {state} after about "
             f"{job['charged']} min of GPU time ({job['source']}). Output: `{job['folder']}/gpu-output/`{note}")
    safefs.append_text(LOG, entry + "\n")
    safefs.append_text(SERVICE_LOG, f"- {stamp()} Kaggle: ran `{job['line']}`, used about {job['charged']} GPU "
                                    f"minutes ({used_this_week(u)} of {WEEKLY_MINUTES} in the last 7 days), "
                                    f"result: {state}. (Logged automatically by the GPU workflow.)\n")
    print(entry)


def wait(job: dict, u: dict, deadline: float) -> bool:
    """Wait for a job; True if it finished, False if it's still running at the deadline."""
    while time.monotonic() < deadline:
        time.sleep(60)  # first, so a just-sent job isn't mistaken for its finished previous version
        state = status(job["kernel"])
        if state.startswith(DONE):
            finish(job, state, u)
            return True
    return False


def main() -> int:
    ledger = ROOT / "state" / "budget.json"
    if ledger.exists() and json.loads(ledger.read_text()).get("paused"):
        print("The agent is paused; GPU jobs don't run.")
        return 0
    if not os.environ.get("KAGGLE_API_TOKEN"):
        print("No KAGGLE_API_TOKEN secret; nothing sent.")
        return 0
    deadline = time.monotonic() + POLL_MINUTES * 60
    u = load_usage()

    for job in [j for j in u["jobs"] if j.get("pending")]:  # still running from an earlier run
        age = (now() - dt.datetime.fromisoformat(job["submitted"])).total_seconds() / 3600
        if age > GIVE_UP_HOURS:
            # Kaggle stops a job within 12 hours, so its status should have settled by now. Stop
            # waiting on it, or it would hold up every later job.
            finish(job, f"lost (no finished status after {GIVE_UP_HOURS} hours)", u)
            continue
        if not wait(job, u, deadline):
            print(f"{job['kernel']} is still running; the next run will collect it.")
            return 0

    # Read without following a symlink: a queue pointing at this process's own environment
    # would otherwise put the Kaggle key into the log as an unreadable job line.
    queue = safefs.read_text(QUEUE)
    lines = [l.strip() for l in queue.splitlines()] if queue is not None else []
    # Skip lines already taken that the session hasn't removed from the queue yet.
    try:
        done = set(json.loads(QUEUE_DONE.read_text())["cleared"])
    except Exception:
        done = set()
    taken = Counter(j["line"] for j in u["jobs"] if j.get("from_queue") and gpu_job_id(j) not in done)
    jobs = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if taken[line] > 0:
            taken[line] -= 1
        else:
            jobs.append(line)

    entries = []
    for line in jobs:
        try:
            script, minutes = parse(line)
            minutes = min(minutes, MAX_JOB_MINUTES)
            left = WEEKLY_MINUTES - used_this_week(u)
            if minutes + OVERHEAD > left:
                entries.append(f"- {stamp()} `{line}`: waiting, it needs {minutes + OVERHEAD} GPU minutes "
                               f"and {max(0, left)} of the {WEEKLY_MINUTES} for the last 7 days are left")
                break
            source = provenance(script)
            kernel = submit(script, minutes)
        except Exception as e:
            entries.append(f"- {stamp()} `{line}`: not run ({e})")
            # Recorded as taken, so the session removes it from the queue.
            u["jobs"].append({"line": line, "submitted": now().isoformat(timespec="seconds"), "charged": 0,
                              "pending": False, "status": "not run", "from_queue": True, "id": uuid.uuid4().hex})
            continue
        job = {"line": line, "folder": str(script.parent.relative_to(ROOT)).replace("\\", "/"),
               "kernel": kernel, "limit": minutes, "charged": minutes + OVERHEAD, "source": source,
               "submitted": now().isoformat(timespec="seconds"), "pending": True, "status": "running",
               "from_queue": True, "id": uuid.uuid4().hex}
        u["jobs"].append(job)
        save_usage(u)
        print(f"Sent {line} to Kaggle as {kernel} (limit {minutes} min)")
        if not wait(job, u, deadline):
            entries.append(f"- {stamp()} `{line}`: still running on Kaggle ({kernel}); "
                           "the next GPU run will collect it")
            break

    if entries:
        safefs.append_text(LOG, "\n".join(entries) + "\n")
    u["jobs"] = u["jobs"][-200:]
    save_usage(u)
    return 0


if __name__ == "__main__":
    sys.exit(main())
