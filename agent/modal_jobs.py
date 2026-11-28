"""GPU jobs on Modal: runs the experiments the agent queued for Modal, with no Claude calls.

Works like agent/kaggle_jobs.py. The agent adds lines to self/modal_queue.txt during a session:

    experiments/my-experiment/train.py 120

(a .py or .sh script under experiments/, then a time limit in minutes, at most 300). After the
session, the "Modal GPU jobs" workflow runs each one on Modal with one NVIDIA L4 GPU, one at a
time, waits for it, and saves what it wrote into gpu-output/ next to the script. Results are logged
in self/modal_log.md (its service log too), which the next session reads. Like the Kaggle workflow,
this never edits the queue; it records the lines it took, and the session removes them.

The Modal token stays in this workflow. The agent's code runs only in Modal's container (see
agent/modal_app.py), never here, and only the small files of its own experiment folder are sent.
Use is kept in state/modal_usage.json, in dollars at Modal's list prices, and capped per calendar
month (MODAL_MONTHLY_USD, under the plan's free credit). The workspace budget set on Modal's site is
the hard limit behind this one.
"""

import base64
from collections import Counter
import datetime as dt
import json
import os
import shutil
import sys
import tarfile
import io
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import ROOT, gpu_job_id, parse, provenance  # noqa: E402  (same queue rules as the lab)
from kaggle_jobs import bundle  # noqa: E402  (same packing as Kaggle jobs)
import safefs  # noqa: E402

QUEUE = ROOT / "self" / "modal_queue.txt"
# Its own log, which is also the service log for Modal: the Kaggle workflow may be saving
# gpu_log.md and service_log.md at the same time, and two runs appending to one file would collide.
LOG = ROOT / "self" / "modal_log.md"
USAGE = ROOT / "state" / "modal_usage.json"
QUEUE_DONE = ROOT / "state" / "modal_queue_done.json"   # see QUEUE_DONE in agent/kaggle_jobs.py

MONTHLY_USD = float(os.environ.get("MODAL_MONTHLY_USD") or 28)   # of the $30 credit; Modal stops all use at $29.50
RUN_MINUTES = int(os.environ.get("MODAL_RUN_MINUTES") or 330)     # how long one workflow run may work
OVERHEAD = 10                                                       # image start-up, per job (minutes)

CHECK_SCRIPT = r'''import subprocess, sys
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
for m in ("vllm", "peft", "trl", "transformers", "datasets"):
    try:
        print(m, __import__(m).__version__)
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, e)
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
    USAGE.parent.mkdir(exist_ok=True)
    tmp = USAGE.with_name(USAGE.name + ".tmp")
    tmp.write_text(json.dumps(u, indent=2) + "\n")
    os.replace(tmp, USAGE)


def used_this_month(u: dict) -> float:
    month = f"{now():%Y-%m}"
    return sum(j["dollars"] for j in u["jobs"] if j["submitted"].startswith(month))


def dollars(minutes: float, price: float) -> float:
    return round(minutes * 60 * price, 2)


def clean(text: str) -> str:
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[key]")
    return text[-600:]


def run_remote(packed: bytes, entry: str, minutes: int) -> bytes:
    """Run one job on Modal and wait for it. Stopping this process stops the job too: the app
    only lives while this runs, so a stopped workflow can't leave a GPU running."""
    import modal
    from modal_app import app, run_job
    with modal.enable_output():
        with app.run():
            return run_job.remote(packed, entry, minutes)


def save_output(result: bytes, folder: str) -> None:
    dest = ROOT / folder / "gpu-output"
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result)) as t:
        t.extractall(dest, filter="data")
    for p in dest.rglob("*"):
        if p.is_symlink():
            p.unlink()


def record(job: dict, state: str, u: dict, minutes: float, price: float, note: str = "") -> None:
    job["dollars"] = dollars(minutes, price)
    job["minutes"] = round(minutes)
    job["status"], job["pending"], job["finished"] = state, False, now().isoformat(timespec="seconds")
    save_usage(u)
    where = f" Output: `{job['folder']}/gpu-output/`" if job.get("folder") else ""
    entry = (f"- {stamp()} `{job['line']}` on Modal (L4): {state} after about {job['minutes']} min, "
             f"about ${job['dollars']:.2f} (${used_this_month(u):.2f} of ${MONTHLY_USD:.0f} this month; "
             f"{job['source']}).{where}{note}")
    safefs.append_text(LOG, entry + "\n")
    print(entry)


def run_one(job: dict, packed: bytes, entry: str, u: dict, price: float) -> None:
    start = time.monotonic()
    try:
        result = run_remote(packed, entry, job["limit"])
    except Exception as e:
        record(job, f"failed on Modal ({type(e).__name__}: {clean(str(e))})", u,
               (time.monotonic() - start) / 60, price)
        return
    note, state = "", "complete"
    if job.get("folder"):
        try:
            save_output(result, job["folder"])
        except Exception as e:
            note = f" (saving the output failed: {type(e).__name__}: {e})"
    else:  # the setup check: print what it found
        with tarfile.open(fileobj=io.BytesIO(result)) as t:
            text = t.extractfile("gpu-output.txt").read().decode(errors="replace")
        print(text)
        if "[exit code 0 after" not in text or "FAILED" in text:
            state = "check failed (its output is in the workflow log)"
    record(job, state, u, (time.monotonic() - start) / 60, price, note)


def check(u: dict, price: float) -> int:
    """A two-minute test of the setup, started by hand from the workflow page."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "check.py").write_text(CHECK_SCRIPT)
        packed = base64.b64decode(bundle(Path(tmp)))
    job = {"line": "(setup check)", "limit": 5, "source": "a built-in check, not an experiment",
           "submitted": now().isoformat(timespec="seconds"), "pending": True, "dollars": dollars(15, price),
           "from_queue": False, "id": uuid.uuid4().hex}
    u["jobs"].append(job)
    save_usage(u)
    run_one(job, packed, "check.py", u, price)
    return 0 if job["status"] == "complete" else 1


def main() -> int:
    ledger = ROOT / "state" / "budget.json"
    if ledger.exists() and json.loads(ledger.read_text()).get("paused"):
        print("The agent is paused; GPU jobs don't run.")
        return 0
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        print("No MODAL_TOKEN_ID and MODAL_TOKEN_SECRET secrets; nothing sent.")
        return 0
    from modal_app import MAX_JOB_MINUTES, PRICE_PER_SECOND as price
    u = load_usage()

    # This workflow waits for each job, so a job still marked running is one whose workflow run was
    # stopped; stopping the run stopped the job too. Charge it its estimate and move on.
    for job in [j for j in u["jobs"] if j.get("pending")]:
        job.update(status="lost (its workflow run stopped before it finished)", pending=False)
        safefs.append_text(LOG, f"- {stamp()} `{job['line']}` on Modal: {job['status']}; no output\n")
    save_usage(u)

    if "--check" in sys.argv:
        return check(u, price)

    queue = safefs.read_text(QUEUE)
    lines = [l.strip() for l in queue.splitlines()] if queue is not None else []
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

    started, entries = time.monotonic(), []
    for line in jobs:
        try:
            script, minutes = parse(line)
            if minutes > MAX_JOB_MINUTES:
                raise ValueError(f"its time limit is over the {MAX_JOB_MINUTES} minutes a Modal job may have")
            cost, left = dollars(minutes + OVERHEAD, price), MONTHLY_USD - used_this_month(u)
            if cost > left:
                entries.append(f"- {stamp()} `{line}` (Modal): waiting, at its time limit it could cost "
                               f"${cost:.2f} and ${max(0, left):.2f} of this month's ${MONTHLY_USD:.0f} is left")
                break
            if (time.monotonic() - started) / 60 + minutes + OVERHEAD > RUN_MINUTES:
                break  # not enough time left in this run; the next run takes it
            folder = script.parent
            packed = base64.b64decode(bundle(folder))
            source = provenance(script)
        except Exception as e:
            entries.append(f"- {stamp()} `{line}` (Modal): not run ({e})")
            u["jobs"].append({"line": line, "submitted": now().isoformat(timespec="seconds"), "dollars": 0,
                              "pending": False, "status": "not run", "from_queue": True, "id": uuid.uuid4().hex})
            save_usage(u)
            continue
        job = {"line": line, "folder": str(folder.relative_to(ROOT)).replace("\\", "/"), "limit": minutes,
               "source": source, "submitted": now().isoformat(timespec="seconds"), "pending": True,
               "dollars": cost, "status": "running", "from_queue": True, "id": uuid.uuid4().hex}
        u["jobs"].append(job)
        save_usage(u)
        print(f"Running {line} on Modal (limit {minutes} min)", flush=True)
        run_one(job, packed, str(script.relative_to(folder)), u, price)

    if entries:
        safefs.append_text(LOG, "\n".join(entries) + "\n")
    u["jobs"] = u["jobs"][-200:]
    save_usage(u)
    return 0


if __name__ == "__main__":
    sys.exit(main())
