"""One research session of the autonomous AGI research agent.

Runs on a schedule (GitHub Actions). Each session:
  1. works out how much money this session may spend (monthly cap, spread evenly),
  2. gives Claude its charter, its notes, and any messages from the human,
  3. lets Claude research, write notes and run experiments until it ends the session
     or runs out of budget,
  4. records a journal entry and the spend.

Files the agent may not change (its charter, this code, the schedule, the budget
ledger) are restored by the workflow after every session.
"""

import calendar
import datetime as dt
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import anthropic

from lab import PREDICTION  # one definition of what counts as a written prediction

ROOT = Path(__file__).resolve().parent.parent
PROTECTED = ("agent", ".github", "state", ".git", "requirements.txt", "reviews")
STATE_FILE = ROOT / "state" / "budget.json"
# When each prediction was written and each experiment file appeared, recorded by this code
# (the agent can't edit state/), so a reviewer can check predictions came before results.
PROVENANCE_LOG = ROOT / "state" / "provenance.log"
JOURNAL_DIR = ROOT / "research" / "journal"

# The agent's own folder: changes here take effect next session without approval.
SELF_DIR = ROOT / "self"

# Everything a session loads at its start is capped, so memory can't bloat unnoticed: the
# agent is warned as a part nears its cap, and must condense it once it's over.
PLAYBOOK_LIMIT = 10_000
STARTUP_LIMIT = 40_000
ROADMAP_LIMIT = 4_000
AGENDA_LIMIT = 8_000
JOURNAL_ENTRY_LIMIT = 4_000
size_notes: list[str] = []

# The monthly memory tidy-up; its date is kept in this file by the agent.
MEMORY_LOG = SELF_DIR / "memory_log.md"
TIDY_DAYS = 30

# Changes here need a pull request the human approves.
PROPOSABLE = ("agent/", "requirements.txt")
MAX_OPEN_PRS = 3

# "or" rather than a default: GitHub passes unset repository variables as empty strings.
MODEL = os.environ.get("AGENT_MODEL") or "claude-opus-5-5"
EFFORT = os.environ.get("AGENT_EFFORT") or "high"
MONTHLY_BUDGET = float(os.environ.get("MONTHLY_BUDGET_USD") or 40)
RUNS_PER_DAY = int(os.environ.get("RUNS_PER_DAY") or 1)
MIN_SESSION_BUDGET = 0.30
MAX_TURNS = 80
MAX_SECONDS = 45 * 60
MAX_TOOL_OUTPUT = 20_000

# Wind-down: the agent doesn't run unattended forever. After WIND_DOWN_DAYS with no word
# from the human, or when the optional lifetime fund is nearly spent, it writes a final
# report and pauses until the human is back.
WIND_DOWN_DAYS = int(os.environ.get("WIND_DOWN_DAYS") or 180)
WARNING_DAYS = 30
TOTAL_FUNDS = float(os.environ.get("TOTAL_FUNDS_USD") or 0)  # 0 means no lifetime limit
FINAL_REPORT = "research/FINAL_REPORT.md"
# Added to everything the agent posts on GitHub, so its own posts never count as the human
# being in contact (when self-hosted, it posts with the owner's token).
AGENT_MARK = "\n\n<!-- posted by the agent -->"
# GitHub usernames allowed to direct the agent, comma-separated (for example "you,your-backup").
TRUSTED_USERS = {u.strip().lower() for u in os.environ.get("AGENT_TRUSTED_USERS", "").split(",") if u.strip()}

# In GitHub Actions the agent's commands run as a separate user, so they can't read this
# process's memory or environment (which hold the API key and GitHub token) or touch .git.
RUN_AS = os.environ.get("AGENT_RUN_AS")
CACHE_DIR = os.environ.get("AGENT_CACHE_DIR")

# USD per million tokens, and per web search.
PRICES = {
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    # Haiku's price for prompts up to 100K tokens; longer ones cost more, so they're counted at the higher rate.
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-haiku-5-5-long": {"in": 0.50, "out": 2.50, "cache_read": 0.05, "cache_write": 0.625},
}
WEB_SEARCH_PRICE = 0.01


# ---------------------------------------------------------------- budget

def load_ledger(today: dt.date) -> dict:
    month = today.strftime("%Y-%m")
    ledger = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    ledger.setdefault("lifetime_spent_usd", 0.0)
    ledger.setdefault("lifetime_sessions", 0)
    if ledger.get("month") != month:
        ledger.update(month=month, spent_usd=0.0, sessions=0)
    return ledger


def save_ledger(ledger: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(ledger, indent=2) + "\n")


def session_budget(ledger: dict, today: dt.date) -> float:
    remaining = MONTHLY_BUDGET - ledger["spent_usd"]
    if remaining < 0.05:
        return 0.0
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    even_share = remaining / (days_left * RUNS_PER_DAY)
    return min(remaining, max(even_share, MIN_SESSION_BUDGET))


def call_cost(usage) -> float:
    p = PRICES.get(MODEL, PRICES["claude-opus-5-5"])
    prompt = (usage.input_tokens + (getattr(usage, "cache_read_input_tokens", 0) or 0)
              + (getattr(usage, "cache_creation_input_tokens", 0) or 0))
    if MODEL == "claude-haiku-5-5" and prompt > 100_000:
        p = PRICES["claude-haiku-5-5-long"]
    cost = (usage.input_tokens * p["in"]
            + usage.output_tokens * p["out"]
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * p["cache_read"]
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * p["cache_write"]) / 1e6
    server = getattr(usage, "server_tool_use", None)
    if server is not None:
        cost += (getattr(server, "web_search_requests", 0) or 0) * WEB_SEARCH_PRICE
    return cost


# ---------------------------------------------------------------- GitHub issues

def github(method: str, path: str, body: dict | None = None):
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise RuntimeError("GitHub is not available (not running in GitHub Actions)")
    api = os.environ.get("GITHUB_API_URL") or "https://api.github.com"
    req = urllib.request.Request(
        f"{api}/repos/{repo}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or "null")


def trusted(item: dict) -> bool:
    """Written by the human (the GitHub users listed in AGENT_TRUSTED_USERS) or by the agent itself.

    The repository is public, so anyone can comment; only these people may instruct the agent.
    Without a list, falls back to GitHub's owner/member/collaborator labels.
    """
    login = item.get("user", {}).get("login", "")
    if login == "github-actions[bot]":
        return True
    if TRUSTED_USERS:
        return login.lower() in TRUSTED_USERS
    return item.get("author_association") in ("OWNER", "MEMBER", "COLLABORATOR")


def by_agent(item: dict) -> bool:
    return (item.get("user", {}).get("login") == "github-actions[bot]"
            or AGENT_MARK.strip() in (item.get("body") or ""))


def by_human(item: dict) -> bool:
    return trusted(item) and not by_agent(item)


def last_human_contact() -> dt.datetime | None:
    """The latest sign of the human: a commit, issue, pull request or comment of theirs."""
    times = []
    for c in github("GET", "/commits?per_page=30") or []:
        if c["commit"]["author"]["name"] != "agi-research-agent":
            times.append(c["commit"]["committer"]["date"])
            break
    for item in github("GET", "/issues?state=all&sort=created&direction=desc&per_page=30") or []:
        if by_human(item):
            times.append(item["created_at"])
            break
    for c in github("GET", "/issues/comments?sort=created&direction=desc&per_page=50") or []:
        if by_human(c):
            times.append(c["created_at"])
            break
    return max((dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times), default=None)


def human_messages() -> str:
    """Open issues and pull requests labelled 'agent', with their trusted comments."""
    try:
        issues = github("GET", "/issues?labels=agent&state=open&per_page=20")
    except Exception as e:
        return f"(Could not read GitHub issues: {e})"
    issues = [i for i in issues or [] if trusted(i)]
    if not issues:
        return "(No open issues.)"
    parts = []
    for issue in issues:
        kind = "Pull request" if "pull_request" in issue else "Issue"
        text = f"### {kind} #{issue['number']}: {issue['title']}\n{issue.get('body') or ''}\n"
        comments = github("GET", f"/issues/{issue['number']}/comments?per_page=30") or []
        for c in comments:
            if trusted(c):
                who = "you (the agent)" if by_agent(c) else c["user"]["login"]
                text += f"\n**{who}:** {c['body'].replace(AGENT_MARK.strip(), '').strip()}\n"
        hidden = sum(not trusted(c) for c in comments)
        if hidden:
            text += f"\n({hidden} comment(s) from people outside the project were left out.)\n"
        parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------- tools

def resolve(path: str, writing: bool = False) -> Path:
    p = (ROOT / path).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError("path is outside the repository")
    if writing and p.relative_to(ROOT).parts[:1] and p.relative_to(ROOT).parts[0] in PROTECTED:
        raise ValueError(f"{path} is protected; ask the human with request_human_help to change it")
    return p


def clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    return f"{text[:half]}\n\n[... {len(text) - MAX_TOOL_OUTPUT} characters cut ...]\n\n{text[-half:]}"


def tool_read_file(path: str) -> str:
    return clip(resolve(path).read_text(errors="replace"))


def tool_write_file(path: str, content: str) -> str:
    p = resolve(path, writing=True)
    new_dirs = [d for d in reversed(p.parents) if not d.exists()]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    # Keep it editable by the separate user the agent's commands run as.
    for d in new_dirs:
        d.chmod(0o777)
    if p.stat().st_uid == os.getuid():
        p.chmod(0o666)
    return f"Wrote {len(content)} characters to {path}"


def tool_list_dir(path: str = ".") -> str:
    p = resolve(path)
    entries = []
    for child in sorted(p.iterdir()):
        if child.name == ".git":
            continue
        entries.append(child.name + ("/" if child.is_dir() else f"  ({child.stat().st_size} bytes)"))
    return "\n".join(entries) or "(empty)"


def tool_run_command(command: str, timeout_seconds: int = 300) -> str:
    timeout_seconds = max(1, min(int(timeout_seconds), 900))
    # The agent's commands never see the API key or the GitHub token.
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV")}
    if CACHE_DIR:
        # Downloaded models and packages persist between sessions here.
        env.update(XDG_CACHE_HOME=CACHE_DIR, HF_HOME=f"{CACHE_DIR}/huggingface", PIP_CACHE_DIR=f"{CACHE_DIR}/pip")
    argv = ["bash", "-c", f"umask 000; {command}"]
    if RUN_AS:
        home = f"/home/{RUN_AS}"
        env.update(HOME=home, PATH=f"{home}/.local/bin:{env.get('PATH', '')}")
        argv = ["sudo", "-n", "-u", RUN_AS, "env", "-i", *(f"{k}={v}" for k, v in env.items()), *argv]
    # Output goes to temporary files, not pipes: a job the agent starts in the background
    # keeps a pipe open, which would make this wait until the timeout.
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        try:
            r = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err, text=True,
                               timeout=timeout_seconds)
            status = f"exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            status = f"Timed out after {timeout_seconds}s"
        out.seek(0)
        err.seek(0)
        return clip(f"{status}\n--- stdout ---\n{out.read()}\n--- stderr ---\n{err.read()}")


def tool_request_human_help(title: str, body: str) -> str:
    issue = github("POST", "/issues", {"title": title, "body": body + AGENT_MARK, "labels": ["agent"]})
    return f"Opened issue #{issue['number']}. The human's reply will appear in a later session."


def tool_close_issue(number: int, comment: str) -> str:
    github("POST", f"/issues/{number}/comments", {"body": comment + AGENT_MARK})
    github("PATCH", f"/issues/{number}", {"state": "closed"})
    return f"Closed issue #{number}"


def core_rules(charter: str) -> str:
    """The charter's "Core rules" section, which no proposal may change. Working rules may change."""
    start = charter.find("## Core rules")
    if start < 0:
        return ""
    end = charter.find("\n## ", start + 1)
    return charter[start:] if end < 0 else charter[start:end]


# Read once at startup, before the agent can touch anything, so a proposal is always
# checked against the real rules.
CORE_RULES = core_rules((ROOT / "agent" / "charter.md").read_text())
assert CORE_RULES, "agent/charter.md must have a '## Core rules' section"


def tool_propose_change(title: str, description: str, files: list) -> str:
    changes = {}
    for f in files:
        path = posixpath.normpath(f["path"].replace("\\", "/"))
        if path.startswith(("/", "..")):
            raise ValueError(f"{f['path']}: path is outside the repository")
        if not any(path == p or path.startswith(p) for p in PROPOSABLE):
            raise ValueError(f"{path}: only files under agent/ and requirements.txt need a pull request. "
                             "Change your other files directly.")
        resolve(path)
        changes[path] = f["content"]
    if "agent/charter.md" in changes:
        if core_rules(changes["agent/charter.md"]) != CORE_RULES:
            raise ValueError("The core rules can't be changed, even in a proposal. "
                             "Working rules and the rest of the charter can.")

    open_prs = [i for i in github("GET", "/issues?labels=agent&state=open&per_page=50") if "pull_request" in i]
    if len(open_prs) >= MAX_OPEN_PRS:
        raise ValueError(f"You already have {len(open_prs)} open pull requests. "
                         "Wait for the human to review them first.")

    base = github("GET", "")["default_branch"]
    base_sha = github("GET", f"/git/ref/heads/{base}")["object"]["sha"]
    base_tree = github("GET", f"/git/commits/{base_sha}")["tree"]["sha"]
    tree = github("POST", "/git/trees", {"base_tree": base_tree, "tree": [
        {"path": p, "mode": "100644", "type": "blob", "content": c} for p, c in changes.items()]})
    commit = github("POST", "/git/commits", {"message": title, "tree": tree["sha"], "parents": [base_sha]})
    branch = f"agent/{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S-%f}"
    github("POST", "/git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    pr = github("POST", "/pulls", {"title": title, "head": branch, "base": base, "body": description + AGENT_MARK})
    github("POST", f"/issues/{pr['number']}/labels", {"labels": ["agent"]})
    return f"Opened pull request #{pr['number']}. It takes effect only after the human approves and merges it."


HANDLERS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
    "run_command": tool_run_command,
    "request_human_help": tool_request_human_help,
    "close_issue": tool_close_issue,
    "propose_change": tool_propose_change,
}


def schema(props: dict, required: list) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 15},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 20},
    {"name": "read_file", "description": "Read a text file in the repository.",
     "input_schema": schema({"path": {"type": "string", "description": "Path relative to the repository root."}}, ["path"])},
    {"name": "write_file", "description": "Create or overwrite a text file in the repository. "
                                          "Files under agent/, .github/ and state/ are protected.",
     "input_schema": schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])},
    {"name": "list_dir", "description": "List a directory in the repository.",
     "input_schema": schema({"path": {"type": "string", "description": "Defaults to the repository root."}}, [])},
    {"name": "run_command", "description": "Run a shell command (bash) from the repository root on the "
                                           "Linux machine running this session: run experiments, pip install packages, "
                                           "inspect output. 2 CPU cores, ~7 GB RAM, no GPU. Max 900 s per command.",
     "input_schema": schema({"command": {"type": "string"},
                             "timeout_seconds": {"type": "integer", "description": "Default 300, max 900."}}, ["command"])},
    {"name": "request_human_help", "description": "Open a GitHub issue asking the human for something you cannot "
                                                  "do yourself. Replies arrive in a later session.",
     "input_schema": schema({"title": {"type": "string"}, "body": {"type": "string"}}, ["title", "body"])},
    {"name": "close_issue", "description": "Comment on and close a GitHub issue that has been dealt with.",
     "input_schema": schema({"number": {"type": "integer"}, "comment": {"type": "string"}}, ["number", "comment"])},
    {"name": "propose_change", "description": "Propose changes to your own code (agent/) or requirements.txt as a "
                                              "pull request for the human to review. Give the full new content of "
                                              "each changed file. Nothing changes until the human merges it.",
     "input_schema": schema({
         "title": {"type": "string"},
         "description": {"type": "string", "description": "What changes, why, and how you tested it."},
         "files": {"type": "array", "items": schema({"path": {"type": "string"}, "content": {"type": "string"}},
                                                    ["path", "content"])},
     }, ["title", "description", "files"])},
    {"name": "end_session", "description": "End this session. Call it once your work for the session is "
                                           "recorded in files.",
     "input_schema": schema({
         "summary": {"type": "string", "description": "What you did, what you found, and what the next session "
                                                      "should do. Saved as this session's journal entry. Keep it "
                                                      "under about 2,500 characters."},
         "new_result": {"type": "string", "description": "The new result this session produced, in one sentence "
                                                         "(a finished experiment, a measured comparison, or a "
                                                         "conclusion backed by evidence). Empty if none."},
     }, ["summary", "new_result"])},
]


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    try:
        return str(HANDLERS[name](**args)), False
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", True


# ---------------------------------------------------------------- session

def prediction_fingerprint(text: str) -> str | None:
    """A short hash of a README's Prediction section, or None if it has none."""
    found = PREDICTION.search(text)
    if not found:
        return None
    rest = text[found.start():]
    next_heading = re.search(r"\n\s*#+\s", rest[1:])
    section = rest[:next_heading.start() + 1] if next_heading else rest
    return hashlib.sha256(section.strip().encode()).hexdigest()[:12]


def scan_experiments() -> dict[str, str]:
    """Every file under experiments/: READMEs by their prediction's fingerprint, others as 'file'."""
    found = {}
    for p in (ROOT / "experiments").rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if p.name == "README.md":
            try:
                fp = prediction_fingerprint(p.read_text(errors="replace"))
            except OSError:
                fp = None
            found[rel] = f"prediction {fp}" if fp else "readme"
        else:
            found[rel] = "file"
    return found


def record_provenance(seen: dict[str, str], events: list[str]) -> dict[str, str]:
    """Log predictions written or changed, and experiment files created, since the last scan."""
    now = scan_experiments()
    stamp = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}"
    for path, kind in now.items():
        before = seen.get(path)
        if kind.startswith("prediction") and before != kind:
            what = "prediction changed" if before and before.startswith("prediction") else "prediction written"
            events.append(f"{stamp} {what}: {path} ({kind.split()[1]})")
        elif kind == "file" and before is None:
            events.append(f"{stamp} file created: {path}")
    return now


def capped(label: str, text: str, limit: int) -> str:
    """Cut text to its limit, and note for the agent when it's over or close to it."""
    if len(text) > limit:
        size_notes.append(f"- **{label} is {len(text):,} characters, over its {limit:,} limit**, so only the "
                          f"first {limit:,} were loaded. Condense it this session.")
        return text[:limit] + f"\n\n[... cut at the {limit:,}-character limit ...]"
    if len(text) > 0.8 * limit:
        size_notes.append(f"- {label} is {len(text):,} of its {limit:,} characters. Condense it soon.")
    return text


def days_since_date(text: str, label: str, now: dt.datetime) -> int | None:
    """Days since a 'Label: YYYY-MM-DD' line in text, or None if there's no valid one."""
    found = re.search(rf"{label}:\**\s*(\d{{4}}-\d{{2}}-\d{{2}})", text)
    try:
        return (now.date() - dt.date.fromisoformat(found.group(1))).days if found else None
    except ValueError:
        return None


def system_prompt() -> str:
    charter = (ROOT / "agent" / "charter.md").read_text()
    playbook = SELF_DIR / "playbook.md"
    if not playbook.exists():
        return charter
    text = capped("self/playbook.md", playbook.read_text(errors="replace"), PLAYBOOK_LIMIT)
    return (f"{charter}\n\n---\n\n# Your playbook (self/playbook.md)\n\n"
            "You wrote this yourself in earlier sessions. Where it conflicts with the charter above, "
            f"the charter wins.\n\n{text}")


def startup_files() -> str:
    """Files the agent listed in self/startup.txt, one path per line, loaded at the start of each session."""
    listing = SELF_DIR / "startup.txt"
    if not listing.exists():
        return "(None. List files in self/startup.txt, one per line, to have them loaded here.)"
    parts, total = [], 0
    for line in listing.read_text().splitlines():
        path = line.strip()
        if not path or path.startswith("#"):
            continue
        try:
            text = resolve(path).read_text(errors="replace")
        except Exception as e:
            parts.append(f"### {path}\n(Could not read: {e})")
            continue
        if total + len(text) > STARTUP_LIMIT:
            parts.append(f"### {path}\n(Skipped: the {STARTUP_LIMIT}-character limit for startup files is reached.)")
            size_notes.append(f"- **Startup files are over their {STARTUP_LIMIT:,}-character limit**, so {path} "
                              "was skipped. Condense them or list fewer.")
            continue
        total += len(text)
        parts.append(f"### {path}\n{text}")
    if STARTUP_LIMIT * 0.8 < total <= STARTUP_LIMIT:
        size_notes.append(f"- Startup files are {total:,} of their {STARTUP_LIMIT:,} characters. Condense them soon.")
    return "\n\n".join(parts) or "(self/startup.txt lists no files.)"


def progress_line(ledger: dict, now: dt.datetime) -> str:
    since = ledger.get("sessions_since_result", 0)
    finished = 0
    log = SELF_DIR / "lab_log.md"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                when = dt.datetime.strptime(line[2:18], "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            finished += "exit code 0" in line and (now - when).days < 30
    text = (f"Progress: {since} session(s) since your last new result; {ledger.get('lifetime_results', 0)} "
            f"new results in total; {finished} lab job(s) finished successfully in the last 30 days.")
    if since >= 3:
        text += (f"\n\n**Your last {since} sessions produced no new result.** This session, get one: pick the "
                 "smallest experiment that answers a real question, and run it.")
    return text


def roadmap_section(now: dt.datetime) -> str:
    """The agent's roadmap, plus a reminder when it is missing or its monthly check is due."""
    roadmap = ROOT / "research" / "roadmap.md"
    if not roadmap.exists():
        return ("## Your long-term roadmap (research/roadmap.md)\n\n**It doesn't exist yet.** Write it this "
                "session, briefly: your vision, 3–5 milestones with how each is measured, where you stand, "
                "and `Last checked:` with today's date. See \"Your long-term roadmap\" in your charter.\n")
    text = roadmap.read_text(errors="replace")
    days = days_since_date(text, "Last checked", now)
    due = ""
    if days is None:
        due = "\n\n**Monthly check due:** the roadmap has no valid `Last checked: YYYY-MM-DD` date."
    elif days >= 30:
        due = f"\n\n**Monthly check due:** last checked {days} days ago. Do it this session."
    return (f"## Your long-term roadmap (research/roadmap.md)\n\n"
            f"{capped('research/roadmap.md', text, ROADMAP_LIMIT)}{due}\n")


def journal_entry(path: Path) -> str:
    """A past journal entry without its list of actions, cut to its limit. Entries are never
    edited afterwards, so a long one means the agent should write shorter summaries."""
    text = re.sub(r"<details>.*?</details>", "", path.read_text(errors="replace"), flags=re.S).strip()
    if len(text) > JOURNAL_ENTRY_LIMIT:
        size_notes.append(f"- Journal entry {path.name} is {len(text):,} characters, over its "
                          f"{JOURNAL_ENTRY_LIMIT:,} limit, so it was cut. Keep your end_session summaries shorter.")
        text = text[:JOURNAL_ENTRY_LIMIT] + "\n\n[... cut ...]"
    return text


def memory_tidy_section(now: dt.datetime) -> str:
    """A reminder when the monthly memory tidy-up is due; nothing otherwise."""
    text = MEMORY_LOG.read_text(errors="replace") if MEMORY_LOG.exists() else ""
    days = days_since_date(text, "Last tidied", now)
    if days is not None and days < TIDY_DAYS:
        return ""
    when = f"last done {days} days ago" if days is not None else "no valid `Last tidied:` date found"
    return (f"## Memory tidy-up due ({when})\n\nThis session, tidy your memory as described in your charter "
            f"(\"Keeping your memory healthy\"), then update `Last tidied:` in `self/memory_log.md`.\n")


def memory_size_section() -> str:
    """Warnings collected while loading this session's memory; nothing if all is within limits."""
    if not size_notes:
        return ""
    return "## Memory size\n\n" + "\n".join(size_notes) + "\n"


def latest_review(now: dt.datetime) -> str:
    """This week's independent review, if there is one."""
    reviews = sorted((ROOT / "reviews").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md"))
    if not reviews:
        return ""
    try:
        when = dt.datetime.strptime(reviews[-1].stem, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return ""
    if now - when > dt.timedelta(days=7):
        return ""
    return (f"## This week's independent review (reviews/{reviews[-1].name})\n\n"
            "A fresh reviewer with no stake in your work checked last week's claims. Fix anything under "
            "\"Could change a result\" (or, if you disagree, say why in the experiment's README), and "
            "relabel any claim it found overstated. Note the minor points and move on.\n\n"
            f"{reviews[-1].read_text(errors='replace')[:10_000]}\n")


def lab_status() -> str:
    log, queue = SELF_DIR / "lab_log.md", SELF_DIR / "lab_queue.txt"
    text = log.read_text(errors="replace")[-4000:] if log.exists() else "(The lab hasn't run anything yet.)"
    waiting = [l for l in queue.read_text().splitlines() if l.strip() and not l.startswith("#")] if queue.exists() else []
    return f"{text}\n\nStill queued: {len(waiting)} job(s)."


FINAL_SESSION_NOTE = f"""## This is your final session before pausing

{{reason}} Use this session to wind down well:

1. Write `{FINAL_REPORT}`: what you set out to do, what you tried, what you found (including what failed), how confident you are in each finding, what is unfinished, and where someone picking this up should start. Write it for a reader who has never seen this repository.
2. Update `research/agenda.md` and `knowledge/index.md` so they are accurate.
3. Don't queue lab jobs or open pull requests; nothing will review them.
4. Call `end_session` with a short summary.

After this session you will pause. You'll start again on your own as soon as the human is back."""


def first_message(budget: float, ledger: dict, now: dt.datetime, extra: str = "") -> str:
    agenda = ROOT / "research" / "agenda.md"
    journals = sorted(JOURNAL_DIR.glob("*.md"))
    recent = "\n\n---\n\n".join(journal_entry(j) for j in journals[-3:]) or "(No earlier sessions.)"
    agenda_text = (capped("research/agenda.md", agenda.read_text(errors="replace"), AGENDA_LIMIT)
                   if agenda.exists() else "(missing - create it)")
    contact = ledger.get("last_human_contact")
    contact_line = (f"Last contact from the human: {(now - dt.datetime.fromisoformat(contact)).days} days ago. "
                    f"After {WIND_DOWN_DAYS} days without contact you write a final report and pause."
                    if contact else "Last contact from the human: unknown.")
    funds_line = (f" Lifetime fund: ${ledger['lifetime_spent_usd']:.2f} of ${TOTAL_FUNDS:.2f} spent."
                  if TOTAL_FUNDS else "")
    return f"""Session started {now:%Y-%m-%d %H:%M} UTC. This is session #{ledger['lifetime_sessions'] + 1}.

Budget for this session: ${budget:.2f}. Spent so far this month: ${ledger['spent_usd']:.2f} of ${MONTHLY_BUDGET:.2f}.{funds_line}
{contact_line}
{progress_line(ledger, now)}
{extra}

{roadmap_section(now)}
## Your research agenda (research/agenda.md)
{agenda_text}

## Your last journal entries
{recent}

{latest_review(now)}
## Recent lab runs (self/lab_log.md)
{lab_status()}

## Files you chose to load at startup (self/startup.txt)
{startup_files()}

## Open GitHub issues and pull requests (labelled 'agent')
{human_messages()}

{memory_tidy_section(now)}
{memory_size_section()}
Begin."""



def run_session(client, charter: str, messages: list, budget: float, s: dict) -> str:
    """The agent loop. Updates s["spent"], s["turns"] and s["log"] as it goes; returns the summary."""
    started = time.monotonic()
    warned = False
    for s["turns"] in range(1, MAX_TURNS + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=charter,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default", "cache_control": {"type": "ephemeral"}},
        )
        s["spent"] += call_cost(response.usage)
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[claude] {block.text.strip()}")
            elif block.type == "server_tool_use":
                s["log"].append(f"{block.name}: {json.dumps(block.input)[:200]}")
                print(f"[{block.name}] {json.dumps(block.input)[:200]}")

        if response.stop_reason == "refusal":
            return f"Session stopped: the model declined to continue ({response.stop_details})."
        if s["spent"] >= budget:
            return "(Budget ran out before the agent wrote a summary.)"
        if time.monotonic() - started > MAX_SECONDS:
            return "(Time limit reached before the agent wrote a summary.)"
        if response.stop_reason == "pause_turn":
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            messages.append({"role": "user", "content":
                             "Keep working, or call end_session with your summary if you are done."})
            continue

        results, summary = [], None
        for tu in tool_uses:
            s["log"].append(f"{tu.name}: {json.dumps(tu.input)[:200]}")
            print(f"[{tu.name}] {json.dumps(tu.input)[:200]}")
            if tu.name == "end_session":
                summary = tu.input.get("summary", "")
                s["new_result"] = tu.input.get("new_result", "").strip()
                output, is_error = "Session ended.", False
            else:
                output, is_error = run_tool(tu.name, tu.input)
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": output, "is_error": is_error})
        s["seen"] = record_provenance(s["seen"], s["events"])
        if summary is not None:
            return summary

        near_budget = s["spent"] >= 0.8 * budget
        near_time = time.monotonic() - started > 0.8 * MAX_SECONDS
        if (near_budget or near_time) and not warned:
            warned = True
            results.append({"type": "text", "text":
                            f"Notice: ${s['spent']:.2f} of ${budget:.2f} spent, "
                            f"{(time.monotonic() - started) / 60:.0f} of {MAX_SECONDS // 60} minutes used. "
                            "Save your work to files, update research/agenda.md, and call end_session now."})
        messages.append({"role": "user", "content": results})
    return "(Reached the turn limit before the agent wrote a summary.)"


def parse_time(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def post_issue(title: str, body: str) -> None:
    try:
        github("POST", "/issues", {"title": title, "body": body + AGENT_MARK, "labels": ["agent"]})
    except Exception as e:
        print(f"Could not open the issue '{title}': {e}")


def plan_session(ledger: dict, now: dt.datetime, budget: float) -> tuple[str, str, str]:
    """Decide what this session is: 'normal', 'final' (write the final report, then pause)
    or 'skip' (paused). Returns (mode, note for the agent, reason)."""
    try:
        contact = last_human_contact()
    except Exception as e:
        print(f"Could not check for contact from the human: {e}")
        contact = None
    if ledger.get("last_human_contact"):
        stored = parse_time(ledger["last_human_contact"])
        contact = max(contact, stored) if contact else stored
    if contact:
        ledger["last_human_contact"] = contact.isoformat()
    funds_left = TOTAL_FUNDS - ledger["lifetime_spent_usd"] if TOTAL_FUNDS else None

    if ledger.get("paused"):
        if ledger.get("pause_reason") == "funds":
            back = funds_left is None or funds_left > 2 * budget
        else:
            back = contact is not None and contact > parse_time(ledger["paused_at"])
        if not back:
            return "skip", "", "The agent is paused until the human is back."
        for key in ("paused", "paused_at", "pause_reason", "warned_inactive"):
            ledger.pop(key, None)
        return "normal", ("## Welcome back\n\nYou paused after writing your final report "
                          f"(`{FINAL_REPORT}`), and the human has been in contact since. Read the report "
                          "and any new messages, then carry on."), ""

    if funds_left is not None and funds_left <= 0.05:
        return "skip", "", "The lifetime fund is used up."
    if funds_left is not None and funds_left <= 2 * budget:
        return "final", FINAL_SESSION_NOTE.format(
            reason=f"The fund the human set aside is almost used up (${funds_left:.2f} left)."), "funds"
    if contact is None:
        return "normal", "", ""

    days = (now - contact).days
    if days >= WIND_DOWN_DAYS:
        return "final", FINAL_SESSION_NOTE.format(
            reason=f"The human hasn't been in contact for {days} days."), "inactive"
    if days >= WIND_DOWN_DAYS - WARNING_DAYS:
        if not ledger.get("warned_inactive"):
            post_issue("Are you still there?",
                       f"I haven't heard from you in {days} days. If I don't hear from you within "
                       f"{WIND_DOWN_DAYS - days} days, I'll write a final report of my work and pause. "
                       "Any comment here keeps me going.")
            ledger["warned_inactive"] = True
        return "normal", (f"## Note\n\nThe human hasn't been in contact for {days} days. If that lasts "
                          f"another {WIND_DOWN_DAYS - days} days, you'll write a final report and pause. "
                          "Keep working as usual, and keep your notes in good order."), ""
    ledger.pop("warned_inactive", None)
    return "normal", "", ""


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    ledger = load_ledger(now.date())
    budget = session_budget(ledger, now.date())
    if budget <= 0:
        print(f"Monthly budget of ${MONTHLY_BUDGET:.2f} is used up; skipping this session.")
        return 0

    mode, note, reason = plan_session(ledger, now, budget)
    if mode == "skip":
        print(f"{reason} Skipping this session.")
        save_ledger(ledger)
        return 0
    if TOTAL_FUNDS:
        budget = min(budget, TOTAL_FUNDS - ledger["lifetime_spent_usd"])

    size_notes.clear()
    charter = system_prompt()
    messages = [{"role": "user", "content": first_message(budget, ledger, now, note)}]
    s = {"spent": 0.0, "turns": 0, "log": [], "new_result": "", "seen": scan_experiments(), "events": []}
    summary = "(The session crashed; see the GitHub Actions log.)"
    try:
        summary = run_session(anthropic.Anthropic(), charter, messages, budget, s)
    finally:
        # Always record the spend, even after a crash, and overwrite any change the agent made to the ledger.
        ledger["spent_usd"] = round(ledger["spent_usd"] + s["spent"], 4)
        ledger["lifetime_spent_usd"] = round(ledger["lifetime_spent_usd"] + s["spent"], 4)
        ledger["sessions"] += 1
        ledger["lifetime_sessions"] += 1
        if s["events"]:
            PROVENANCE_LOG.parent.mkdir(exist_ok=True)
            with open(PROVENANCE_LOG, "a") as log:
                log.write(f"# Session {ledger['lifetime_sessions']}\n" + "\n".join(s["events"]) + "\n")
        if s["new_result"]:
            ledger["sessions_since_result"] = 0
            ledger["lifetime_results"] = ledger.get("lifetime_results", 0) + 1
        else:
            ledger["sessions_since_result"] = ledger.get("sessions_since_result", 0) + 1
        if mode == "final":
            ledger.update(paused=True, paused_at=now.isoformat(), pause_reason=reason)
            if reason == "funds":
                why = "The fund you set aside for me is almost used up."
                wake = "Raise `TOTAL_FUNDS_USD` in the repository's Actions variables to restart me."
            else:
                why = f"I haven't heard from you in over {WIND_DOWN_DAYS} days."
                wake = "Comment on this issue, or anywhere in the repository, to restart me."
            post_issue("I've paused",
                       f"{why}\n\nI wrote a final report of my work in `{FINAL_REPORT}` "
                       f"and have stopped running. {wake}\n\nIf GitHub has disabled the scheduled workflows "
                       "after a long quiet period, re-enable them in the Actions tab.")
        save_ledger(ledger)

        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        entry = JOURNAL_DIR / f"{now:%Y-%m-%d-%H%M}.md"
        actions = "\n".join(f"- `{a}`" for a in s["log"]) or "- (none)"
        entry.write_text(f"# Session {ledger['lifetime_sessions']} — {now:%Y-%m-%d %H:%M} UTC\n\n"
                         f"Cost: ${s['spent']:.2f} (budget ${budget:.2f}), turns: {s['turns']}\n\n"
                         f"**New result:** {s['new_result'] or '(none)'}\n\n"
                         f"## Summary\n\n{summary}\n\n"
                         f"<details><summary>Actions</summary>\n\n{actions}\n\n</details>\n")
        print(f"\nSession done. Spent ${s['spent']:.2f}. Journal: {entry.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
