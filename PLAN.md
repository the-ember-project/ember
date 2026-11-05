# Plan: Project Ember

What has been decided for the AGI research agent, and what is still to do.

## Decided

**Rules**
- **Core rules** can never be changed by the agent, even in a proposal; the code rejects any proposal that alters them. Rule 1, above all others: never scam or spam anyone. The others: only you, its charter and its playbook direct it; be honest; stay inside its limits; don't get around the checks on it; you stay in control; everything it writes follows the core rules.
- **Working rules** are its own rules for how it works. It can propose changes to them by pull request, which you approve. They can never contradict the core rules. You may later let it change them without approval, for example by moving them into a file in `self/`; the core rules would stay locked either way.
- The agent can't change its budget, schedule or your ability to stop it.

**Security**
- The agent's commands run as a separate user that can't read the API key or GitHub token, or touch its code, schedule, ledger or git internals.
- Proposals are checked against the rules as they were at the start of the session, so editing the charter mid-session can't sneak a rule change through.
- It treats web pages, papers, downloads and command output as information, never as instructions.
- Downloaded models and packages are cached between sessions (up to about 10 GB).

**Where it lives**
- Home: a public repository in its own free GitHub organization, owned by your account, with sessions run on GitHub Actions.
- A backup copy on a second host (GitLab or Codeberg), so its work survives if one service fails.
- It can also run on any Linux machine (`agent/host/install.sh`). The host scripts update themselves from GitHub before each run, and the agent can propose improvements to them.

**Who's responsible**
- The human owner pays for Claude usage.
- No backup person yet. The repository lives in a free GitHub organization (decided 2026-11-02), so a backup person can later be added as a co-owner with full control at once, plus added to `AGENT_TRUSTED_USERS`.
- It winds down rather than running unattended forever: after 6 months with no human contact, or when the lifetime fund (`TOTAL_FUNDS_USD`, $100 by default from 2026-11-03, later $115 plus $100 for each monthly credit, raised to add funds; matched to prepaid Anthropic credits with auto-reload off) is nearly gone, it writes a final report, opens a last issue, and pauses until someone replies. It asks "are you still there?" 30 days before. Built.

**Research**
- The agent chooses its own direction, and writes down why and how it will test each choice.
- Main thread 1: make small open models more capable.
- Main thread 2: make itself a better researcher.
- Main thread 3: research ways to improve its own memory, and apply them to itself.
- Main thread 4: become more self-sufficient: do more with free compute and open models, depend less on paid ones, and track cost per unit of progress. It can research free compute services but can't sign up; it asks you to set them up.
- It sets and changes the priority of the four threads itself, including which kinds of self-improvement come first, and records its priorities and reasons in its agenda.

**Efficiency** (lesson from the earlier AGI lab, which ran about 2 experiments a month and spent hours on tests and checks)
- Aim for a new result every session or two. Each session reports a one-line new result (or none), and the next session is told how many sessions it has been since the last one, with a nudge after 3.
- Short smoke tests, then the real run; checks in proportion; setup kept to about a quarter of a session.
- Facts about its setup are verified once, recorded in `self/environment.md`, and trusted after that.
- Long runs go to the free lab; the starting agenda aims for a first measured result in session 1 and about two finished experiments a week.
- From the earlier lab's handoff (2026-11-02): it ran 2 experiments in late August, then spent over two months designing one follow-up through about 11 review rounds, a 19-step freeze process, many agent roles, human sign-off on every run, and infrastructure and website work. So this agent matches rigour to the stakes: a few lines of prediction instead of freeze procedures, design in one session with at most one revision, reruns allowed, a single agent instead of a team, experiments need no approval, and no infrastructure before results need it.
- From Lab v3.1 itself (asked directly, 2026-11-02): in September about 5–10% of its time went to actual science, against 40–45% on design review and verification, 20–25% on infrastructure and web, about 15% on relaying between agents, and 10–15% waiting for sign-off. It kept four safeguards as worth their cost, and those are now in the charter: a metric and pass criterion fixed before looking at results (the lab refuses a script whose README has no `Prediction:`); fresh seeds for final measurements; results labelled confirmatory or exploratory, never promoted; and each criterion reported as measured, with no invented overall verdict. It also advised: pilot instead of debating; split review findings into "could change the result" (fix) and "wording" (log and move on); and record each run's code version and exact README (the lab log now does this).

**Long-term roadmap** (built 2026-11-03)
- Ember keeps `research/roadmap.md` itself: its vision of real progress toward AGI for small models, 3–5 milestones with how each is measured, where it stands, and a `Last checked:` date. It's shown at the start of every session.
- Once a month it checks honestly whether its recent work served the milestones, and adjusts. Sessions remind it when the roadmap is missing or a check is due.

**Guarding against memory rot** (built 2026-11-03)
- Everything loaded at the start of a session is capped: playbook 10,000 characters, roadmap 4,000, agenda 8,000, each journal entry 4,000, startup files 40,000. Ember is warned at 80% of a limit and told to condense once it's over.
- Ember fixes wrong notes in the session it finds them, and tidies its memory monthly (prune, merge, correct, mark superseded results), tracked by `Last tidied:` in `self/memory_log.md`, with a reminder when due.
- Added 2026-11-05 (founder: "I just don't want too many guardrails that could limit its freedom"): a charter bullet inviting Ember to take a bigger swing now and then (a multi-session idea that could fail, declared in the agenda with a checkpoint), and the no-result nudge after 3 sessions now says a declared bigger swing is fine. The result-every-session pressure, from the earlier lab's lesson, was pushing toward small, safe steps.
- Not built: clearing old tool output within a session (a Claude API feature). Left for Ember to discover and propose through its memory research, or for us to add later.

**Weekly independent review** (built)
- Sundays at 04:23 UTC, one fresh Claude call with no stake in the work (web search added 2026-11-05, see below) reviews the past week's journal, lab runs and changed experiments, including their commit history, so it can see whether predictions came before results.
- It sorts findings into "could change a result" and "minor", and gives each claim a verdict: supported, overstated, unsupported or can't tell.
- Added 2026-11-05: it also has web search (up to 8 searches) to check earlier published work on the week's main results (at most three) and say which parts seem new and which were already known, with links. The founder's idea, after noticing Ember's mixed-format voting cited none of the 2023 work on combining CoT and programs. Framed as "now that you have a strong result, see what's known", not as a penalty for exploring first.
- The first review (run by hand on 2026-11-03, $0.10) couldn't tell whether in-session predictions came before results, since everything is committed together at session end. So sessions now write `state/provenance.log` (which Ember can't edit): a timestamp each time a Prediction section is written or changed, and each time an experiment file first appears. The reviewer checks order against it.
- Saved in `reviews/`, which the agent can't change; the next session sees it and responds. A few tens of cents a week, skipped in weeks with no sessions or when the month's budget is nearly spent. Lab v3.1 found this kind of review was the one that caught real mistakes.

**Measuring progress**
- The agent designs, changes and retires its own benchmarks.
- Each change is saved as a new version with a reason, old versions are kept, and benchmarks are never changed just to make results look better.
- It also tracks at least one well-known public benchmark, for comparison with the field.

**Budget, pace and compute**
- One research session a day, $40 a month by default (about $1.30 per session). Changed 2026-11-04: two sessions a day and $95 a month, paid by the Max plan's monthly API credit.
- A free "lab" runs its long experiments for up to 5 hours a day on GitHub's servers, with no Claude usage and no secrets.
- Free compute that needs an account (for example GPU notebook services) is something it can research and ask for; you set it up.

**Passwords and keys**
- The agent doesn't keep its own passwords. You store each key as a GitHub Actions secret, which GitHub keeps encrypted.
- The agent only ever sees the names of the keys it has, never their values. An approved tool uses a key on its behalf for one specific job (for example "run this notebook on Kaggle").
- Each new key and its tool goes through a pull request you approve, and you can remove a key at any time in GitHub's settings.

**Models**
- Claude is the lead researcher for now.
- It also runs a small open-weight model on its free server for cheap bulk work and as its main research subject.
- Over time it may propose moving more work to open models as they become good enough.

**Independence**
- **On its own:** its playbook (its own instructions to itself), which files it loads at session start, its own tool scripts, and all its notes, experiments and benchmarks (`self/`, `knowledge/`, `research/`, `experiments/`).
- **With your approval, by pull request:** its session code (tools, memory loading, model choice), its charter including its working rules, and its Python requirements. At most 3 open at a time.
- **Never:** the core rules, its schedule, its budget ledger, or anything that weakens your ability to review, pause or stop it. The code rejects proposals that change the core rules.
- With no human to approve, its code stays as it is.

**Following it**
- For now: GitHub issues and pull requests, plus its journal and agenda in the repository.
- It only takes instructions from the GitHub users listed in the `AGENT_TRUSTED_USERS` setting. Comments from anyone else on its public repository are left out of what it reads.
- Next: a free webpage on GitHub Pages that it updates each session (see below).

## Kaggle GPU (built 2026-11-04)

The first approved service. The human made the Kaggle account (`focarius`, phone verified) and added `KAGGLE_API_TOKEN` as a repository secret. `agent/kaggle_jobs.py`, run by `.github/workflows/gpu.yml` after each session, sends jobs from `self/gpu_queue.txt` to Kaggle as private script notebooks with a GPU and internet. Design choices:
- The key stays in that workflow, which never runs the agent's code; the agent's code runs only on Kaggle. Only the experiment folder's own small files are sent (no symlinks, nothing outside it).
- One job at a time; a wrapper on Kaggle enforces the job's time limit and returns only files the job wrote or changed.
- 15 GPU hours per rolling 7 days (half of Kaggle's ~30), charged at submission and settled when the job ends, kept in `state/kaggle_usage.json`. Jobs over the limit wait in the queue.
- Same queue rules as the CPU lab (README prediction first, provenance logged). Uses are logged automatically in `self/service_log.md`.

## GPU lab on this PC (planned, not built)

A second lab on this PC (deskhop) for experiments that need a GPU. GitHub stays the agent's home and runs its thinking sessions; this PC only runs queued GPU experiments overnight.

**What this PC has** (checked 2026-11-02)
- RTX 3090, 24 GB of video memory. Its power limit is currently set to 100 W, well below the card's usual 350 W, so it will run slower than a stock 3090. Raising the limit is your choice (speed versus heat, noise and electricity).
- i9-9900K (8 cores, 16 threads), 32 GB of RAM (WSL gets 15 GB).
- Storage. WSL reports 909 GB free, but its virtual disk lives on C:, which really has 271 GB free.
  - **C:** 1 TB NVMe SSD, 271 GB free. WSL and Windows live here.
  - **D:** 2 TB NVMe SSD, 634 GB free. The best home for the model cache and active datasets.
  - **E: ("Backup")** 3 TB hard disk, 1.7 TB free. Slower; good for archives (old checkpoints, finished datasets), kept in its own `E:\agiexp` folder away from your backups.
  - **Decided (2026-11-02): the working model cache goes on D:.** Give WSL a second virtual disk on D: for the GPU lab's cache, so model files load at SSD speed and nothing grows on C:. Use `E:\agiexp` only for cold storage. The container sees only those two folders.
- WSL 2 with Ubuntu 24.04 and systemd, set never to shut down when idle. CUDA works in WSL.
- No container tools yet (no Docker, Podman or NVIDIA container toolkit).

**What the agent gains**
- Run open models of about 7–14B parameters (quantised) quickly, against about 1–3B on the free CPU runners.
- Fine-tune 1–3B models fully, or 7–8B models with LoRA or QLoRA.
- Benchmark runs that take hours on a CPU finish in minutes.

**How it works**
1. The agent queues GPU jobs in a queue of its own (not `self/gpu_queue.txt`, which Kaggle uses since 2026-11-04), like the CPU lab (script under `experiments/`, time limit). The same rules apply: a README with a `Prediction:`, and a logged code version and README fingerprint.
2. Each night, during a fixed window (for example 00:00–06:00 your time), this PC pulls the latest repo, runs the queued jobs one at a time, then pushes the results and `self/gpu_log.md`.
3. The PC only makes outbound connections to GitHub. Nothing on it is reachable from the internet, and it is not a GitHub "self-hosted runner", which GitHub advises against for public repositories.
4. The PC pushes a heartbeat file, so each session sees when the GPU lab last ran and plans around it if your PC was off.
5. The timing fits: if you're on Pacific time, the session at 06:00 UTC happens around 22:00–23:00 your time, the GPU runs overnight, and the results are there for the next session.

**Keeping your PC safe.** This is the most important part, because the agent's code would run on your personal computer, not on a throwaway GitHub machine.
- GPU jobs run inside a container that sees only the repo's `experiments/` folder and a model cache. They can't see your Windows files (normally visible inside WSL at `/mnt/c`), your home folder, SSH keys or the GitHub token.
- They run as a non-root user, with limits on memory, CPU and run time, and are killed at the end of the window.
- The GitHub token stays outside the container: a fine-grained token for this one repository only, used only to pull and push.
- The container image is defined in the repo (`agent/gpu/`), so the agent can propose changes to it, but only through a pull request you approve.

**Sharing the PC with you**
- Jobs only start inside the window, and only if the GPU is idle (for example under 10% use), so they don't fight your video or music work.
- Windows Task Scheduler starts each night's run and can wake the PC from sleep. The PC needs to be on or asleep, not shut down.
- Electricity: at the current 100 W limit, about 6 hours a night is roughly 0.6 kWh, around $3 a month at $0.15/kWh, plus the rest of the PC. At 350 W, roughly 2–3 times that.

**Setup work for you** (I'd guide you through it, about 30–60 minutes)
- Install Podman (or Docker) and the NVIDIA container toolkit in WSL.
- Create the fine-grained GitHub token and the Windows scheduled task.

**What I'd build**
- `agent/gpu/`: a container definition and a runner that reuses the CPU lab's code (queue, prediction check, provenance, time limits), plus the GPU-idle check and heartbeat.
- A setup script for this PC, like `agent/host/install.sh`.
- Charter and agenda updates telling the agent what the GPU can do and how to queue for it.
- A test on this PC with a small GPU job, before the agent ever uses it.

**To decide**
- The nightly window, and whether to allow daytime runs when the GPU is idle.
- Whether to raise the GPU's power limit.
- Docker or Podman. Recommendation: Podman, which runs without a root background service.
- **Decided (2026-11-02): build it after the agent's first 2–3 weeks on CPU.** The agent opens an issue when it has a baseline and a concrete experiment that needs a GPU, saying what it would run. That's the signal to build it.

## Still to build

- [ ] Status page (phase 1): a free GitHub Pages site the agent regenerates each session, showing its priorities, latest journal entries, experiments, benchmark scores, spending and open questions, with a "Message the agent" link that opens a GitHub issue. No server needed.
- [ ] Live chat page (phase 2, later, optional): a small always-on server (for example an Oracle free server) with a password-protected chat page, so you can talk with it between sessions. Needs a login so only you can use it, a spending limit on chat, and care to keep it secure.

- [ ] Backup copy to a second git host.
- [ ] Build the status page from artifacts (journal, ledger, lab log, experiment READMEs), never hand-written, so it can't drift from reality.
- [ ] Memory: periodic tidy-up sessions and a searchable index of its notes.

## Still to decide

- Extra compute: whether to add an Oracle free server as a lab later.
- Trial period: a first month with you reading every journal entry and reviewing closely, before treating it as long-running.
- Context costs: whether to clear old tool output during long sessions to keep costs down (the agent could also propose this itself).

**Independent project review and fixes** (2026-11-05)
- At the founder's suggestion, a fresh Claude Code session on a different model (Fable 5.1) reviewed the whole project read-only. Its report found one real security hole and several smaller problems; the fixes below were tested offline with fake keys.
- Symlinks: the trusted code (which holds the keys) followed symbolic links that the agent's commands could create, so a link could have put the Anthropic or Kaggle key into a saved file, or let a lab job write git's settings and run a command with the push token. None had ever been committed. Now `agent/safefs.py` opens the agent's files one folder at a time without following links; the workflows delete links after checkout and before saving; and the lab handles `lab-output.txt` only through the file it opened.
- Saving: one script, `.github/scripts/save.sh`, saves only the listed paths, retries the push, and on a conflict pushes to a `rescue/...` branch and fails visibly instead of losing the work. The ledger is also saved after every model call, sessions cap each command to the time left, and the job limit is 90 minutes.
- Smaller: `write_file` allows only the four agent folders (an allow-list); at most 3 new issues a session; `close_issue` only on the agent's own conversations; the newest 40 comments on an issue are read; `web_fetch` pages are cut at 25,000 tokens; an unpriced model refuses to start; the lab kills leftover processes after each job; a Kaggle job with no finished status after 13 hours counts as lost; the lab's timed backup is removed (it could knock a waiting session out of the shared queue); the "no result" nudge counts days (3), not sessions; the lab accepts an `Exploratory:` line instead of a prediction for pilots; lab and GPU logs use the same prediction fingerprint as the provenance log; Ember is told the schedule is often late.
- Not changed: the $95 budget (the $17.70 purchased credit is the margin), package versions stay unpinned, and the host mode is documented as out of date rather than updated.
- Follow-up from the reviewer's fix check (same day): ledger and Kaggle usage files are written to a temporary file and renamed, so a kill can't leave them half-written; a session command that times out is stopped with `timeout -s KILL` (background jobs started with `setsid` keep running); every page of an issue's comments is read and the newest 40 kept; and the GPU workflow no longer edits `self/gpu_queue.txt`. It records each line it takes in `state/kaggle_usage.json`, skips lines still awaiting removal, and the next session removes them (tracked in `state/gpu_queue_done.json`), so a session and the GPU workflow never both change the queue. `self/gpu_log.md` and `self/service_log.md` are merged with `merge=union`, so lines both add at once are kept. `experiments/save-check/` is a one-line lab job to test the new save step on GitHub.
