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
import re
import subprocess
import sys
import time
from pathlib import Path

import safefs  # symlink-safe reads and writes of the agent's files

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
QUEUE = ROOT / "self" / "lab_queue.txt"
LOG = ROOT / "self" / "lab_log.md"

TOTAL_MINUTES = int(os.environ.get("LAB_TOTAL_MINUTES") or 300)
DEFAULT_JOB_MINUTES = 60
MAX_OUTPUT = 1_000_000
# A "## Prediction" heading or a "Prediction:" line, in any case.
PREDICTION = re.compile(r"^\s*(#+\s*predictions?\b|\**predictions?\**\s*:)", re.IGNORECASE | re.MULTILINE)
# Or, for pilots, data preparation and other runs that test no prediction, an "Exploratory:"
# line saying what the run is for. Its results are then exploratory, by the charter's rules.
EXPLORATORY = re.compile(r"^\s*\**exploratory\**\s*:", re.IGNORECASE | re.MULTILINE)
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
    readme = safefs.read_text(script.parent / "README.md")
    if readme is None or not (PREDICTION.search(readme) or EXPLORATORY.search(readme)):
        raise ValueError("its folder needs a README.md with a Prediction section or line (metric, pass "
                         "criterion and predicted result) written before the run, or, for a pilot or "
                         "other exploratory run, an 'Exploratory:' line saying what it's for")
    minutes = int(parts[1]) if len(parts) > 1 else DEFAULT_JOB_MINUTES
    return script, max(1, minutes)


def prediction_fingerprint(text: str) -> str | None:
    """A short hash of a README's Prediction section, or None if it has none. The session's
    provenance log uses the same one, so a reviewer can match a run to the prediction it tested."""
    found = PREDICTION.search(text)
    if not found:
        return None
    rest = text[found.start():]
    next_heading = re.search(r"\n\s*#+\s", rest[1:])
    section = rest[:next_heading.start() + 1] if next_heading else rest
    return hashlib.sha256(section.strip().encode()).hexdigest()[:12]


def provenance(script: Path) -> str:
    """The code version and the prediction (by its fingerprint) a run used."""
    fp = prediction_fingerprint(safefs.read_text(script.parent / "README.md") or "")
    used = f"prediction {fp}" if fp else "exploratory run, no prediction"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return f"commit {commit}, {used}"


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
    # Opened without following a symlink, and handled only through this open file afterwards, so
    # the job can't point lab-output.txt at another file (such as git's settings) and have this
    # trusted code write there.
    with safefs.open_write(script.parent / "lab-output.txt", mode=0o666, readable=True) as out:
        try:
            r = subprocess.run(argv, cwd=script.parent, env=env, stdout=out, stderr=subprocess.STDOUT,
                               timeout=minutes * 60)
            status = f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            status = f"stopped at its {minutes}-minute limit"
        stop_leftovers()
        if os.fstat(out.fileno()).st_size > MAX_OUTPUT:
            out.seek(0)
            text = out.read()
            out.seek(0)
            out.truncate()
            out.write(text[:MAX_OUTPUT // 2] + b"\n\n[... output cut ...]\n\n" + text[-MAX_OUTPUT // 2:])
    return status, (time.monotonic() - started) / 60


def stop_leftovers() -> None:
    """Stop everything still running as the agent's user. A time limit only stops sudo, not the
    script under it, and a job may leave background processes; either would keep running into
    the next job, or write files while this trusted code works."""
    if not RUN_AS:
        return
    for _ in range(5):
        if subprocess.run(["sudo", "-n", "pkill", "-KILL", "-u", RUN_AS], capture_output=True).returncode != 0:
            return  # 1 means nothing was left to stop
        time.sleep(1)


def main() -> int:
    ledger = ROOT / "state" / "budget.json"
    if ledger.exists() and json.loads(ledger.read_text()).get("paused"):
        print("The agent is paused; the lab doesn't run.")
        return 0
    queue = safefs.read_text(QUEUE)
    if queue is None:
        print("Lab queue is empty.")
        return 0
    lines = [l.strip() for l in queue.splitlines()]
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

    safefs.append_text(LOG, "\n".join(entries) + "\n")
    comments = [l for l in lines if l.startswith("#")]
    safefs.write_text(QUEUE, "".join(f"{l}\n" for l in comments + remaining))
    if remaining:
        print(f"{len(remaining)} job(s) left in the queue for the next lab run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
