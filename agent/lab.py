"""The lab: runs the long experiments the agent queued, for free, with no Claude calls.

The agent adds lines to self/lab_queue.txt during a session:

    experiments/my-experiment/run.sh 120

(a script under experiments/, then an optional time limit in minutes). After the session,
the lab workflow runs each script in turn on a GitHub Actions runner, as the separate
'researcher' user with no secrets, and saves its output to lab-output.txt next to the
script. Results are logged in self/lab_log.md, which the next session reads.
"""

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
QUEUE = ROOT / "self" / "lab_queue.txt"
LOG = ROOT / "self" / "lab_log.md"

TOTAL_MINUTES = int(os.environ.get("LAB_TOTAL_MINUTES") or 300)
DEFAULT_JOB_MINUTES = 60
MAX_OUTPUT = 1_000_000
RUN_AS = os.environ.get("AGENT_RUN_AS")
CACHE_DIR = os.environ.get("AGENT_CACHE_DIR")


def parse(line: str) -> tuple[Path, int]:
    parts = line.split()
    script = (ROOT / parts[0]).resolve()
    if EXPERIMENTS not in script.parents:
        raise ValueError("the script must be under experiments/")
    if not script.is_file():
        raise ValueError("the script does not exist")
    if script.suffix not in (".sh", ".py"):
        raise ValueError("the script must be a .sh or .py file")
    readme = script.parent / "README.md"
    if not readme.is_file() or "prediction:" not in readme.read_text(errors="replace").lower():
        raise ValueError("its folder needs a README.md with a 'Prediction:' (metric, pass criterion and "
                         "predicted result) written before the run")
    minutes = int(parts[1]) if len(parts) > 1 else DEFAULT_JOB_MINUTES
    return script, max(1, minutes)


def provenance(script: Path) -> str:
    """The code version and the exact README (with its prediction) a run used."""
    readme_hash = hashlib.sha256((script.parent / "README.md").read_bytes()).hexdigest()[:12]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return f"commit {commit}, README {readme_hash}"


def run_job(script: Path, minutes: int) -> tuple[str, float]:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")}
    if CACHE_DIR:
        env.update(XDG_CACHE_HOME=CACHE_DIR, HF_HOME=f"{CACHE_DIR}/huggingface", PIP_CACHE_DIR=f"{CACHE_DIR}/pip")
    interpreter = "python3" if script.suffix == ".py" else "bash"
    argv = ["bash", "-c", f'umask 000; exec {interpreter} "$0"', str(script)]
    if RUN_AS:
        home = f"/home/{RUN_AS}"
        env.update(HOME=home, PATH=f"{home}/.local/bin:{env.get('PATH', '')}")
        argv = ["sudo", "-n", "-u", RUN_AS, "env", "-i", *(f"{k}={v}" for k, v in env.items()), *argv]

    started = time.monotonic()
    out_file = script.parent / "lab-output.txt"
    with open(out_file, "w") as out:
        try:
            r = subprocess.run(argv, cwd=script.parent, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=minutes * 60)
            status = f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            status = f"stopped at its {minutes}-minute limit"
    out_file.chmod(0o666)
    if out_file.stat().st_size > MAX_OUTPUT:
        text = out_file.read_bytes()
        out_file.write_bytes(text[:MAX_OUTPUT // 2] + b"\n\n[... output cut ...]\n\n" + text[-MAX_OUTPUT // 2:])
    return status, (time.monotonic() - started) / 60


def main() -> int:
    ledger = ROOT / "state" / "budget.json"
    if ledger.exists() and json.loads(ledger.read_text()).get("paused"):
        print("The agent is paused; the lab doesn't run.")
        return 0
    if not QUEUE.exists():
        print("Lab queue is empty.")
        return 0
    lines = [l.strip() for l in QUEUE.read_text().splitlines()]
    jobs = [l for l in lines if l and not l.startswith("#")]
    if not jobs:
        print("Lab queue is empty.")
        return 0

    deadline = time.monotonic() + TOTAL_MINUTES * 60
    entries, remaining = [], []
    for i, line in enumerate(jobs):
        stamp = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC"
        try:
            script, minutes = parse(line)
        except Exception as e:
            entries.append(f"- {stamp} `{line}`: not run ({e})")
            continue
        left = (deadline - time.monotonic()) / 60
        if left < 1:
            remaining = jobs[i:]
            break
        minutes = min(minutes, int(left))
        print(f"Running {line} (limit {minutes} min)")
        source = provenance(script)
        status, took = run_job(script, minutes)
        rel = script.parent.relative_to(ROOT)
        entries.append(f"- {stamp} `{line}`: {status} after {took:.0f} min ({source}). "
                       f"Output: `{rel}/lab-output.txt`")

    with open(LOG, "a") as log:
        log.write("\n".join(entries) + "\n")
    comments = [l for l in lines if l.startswith("#")]
    QUEUE.write_text("".join(f"{l}\n" for l in comments + remaining))
    if remaining:
        print(f"{len(remaining)} job(s) left in the queue for the next lab run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
