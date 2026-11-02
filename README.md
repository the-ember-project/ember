# Project Ember

An experiment: Ember, an AI researcher that works on its own, once a day, toward artificial general intelligence. It runs for free on GitHub Actions and uses Claude through the Anthropic API, with a hard monthly spending cap.

Each session it reads its notes, chooses what to work on, searches and reads the web, runs small experiments, writes down what it learned, and plans the next session. Everything it does is committed to this repository, so you can follow it here.

## What's where

| Path | What it is | Who changes it |
|---|---|---|
| `research/agenda.md` | The agent's current plan | Agent |
| `research/journal/` | One entry per session: summary, cost, actions | Agent (written automatically) |
| `knowledge/` | The agent's notes on papers and ideas | Agent |
| `experiments/` | Experiment code and results | Agent |
| `self/` | The agent's playbook, startup memory list, tools and lab queue | Agent |
| `reviews/` | Weekly independent reviews of its work | Weekly review only |
| `agent/charter.md` | The agent's goal and rules | Agent proposes, you approve (never the core rules) |
| `agent/run.py` | The code that runs a session | Agent proposes, you approve |
| `.github/workflows/research.yml` | The daily schedule | You only |
| `state/budget.json` | Spending ledger | `run.py` only |

The agent can't change its code, charter, schedule or ledger directly; any direct change there is undone after each session. To change its own code or charter, it opens a pull request. Nothing changes until you review and merge it. It can't merge pull requests itself, and it can't propose changes to the core rules in its charter, its schedule or its budget. It can propose changes to its working rules.

To change the core rules yourself, edit the "Core rules" section of `agent/charter.md` directly and push.

## Setup (about 15 minutes)

1. **Create a GitHub organization for the agent.** From your GitHub account, go to github.com/organizations/plan and choose the free plan. This gives the agent its own identity, separate from your personal profile. Later you can add a backup person as a co-owner, so the project never depends on you alone.
2. **Create a public repository in that organization**, for example `your-org/agi-agent`. Public repositories get unlimited free Actions minutes.
3. **Push this folder to it:**
   ```bash
   git init -b main
   git add -A
   git commit -m "Start the AGI research agent"
   git remote add origin https://github.com/the-ember-project/ember.git
   git push -u origin main
   ```
4. **Add your API key.** In the repository, go to Settings → Secrets and variables → Actions → New repository secret. Name it `ANTHROPIC_API_KEY` and paste your key.
5. **Say who may direct the agent.** In Settings → Secrets and variables → Actions → Variables, add `AGENT_TRUSTED_USERS` with your GitHub username (add more later, separated by commas). The agent only takes direction from these people; everyone else's comments are ignored.
6. **Set a hard limit on the Anthropic side.** In the Anthropic Console (console.anthropic.com), go to Settings → Limits and set a monthly spend limit. This is your real safety net if anything goes wrong.
7. **Let it open pull requests.** First in the organization's settings, then in the repository's: Settings → Actions → General → "Workflow permissions" → tick "Allow GitHub Actions to create and approve pull requests". (Despite the name, the agent has no tool to approve or merge them.)
8. **Optional: change the monthly budget.** It's $40 by default, which gives each daily session about $1.30. To change it, add a variable `MONTHLY_BUDGET_USD`.
9. **Start the first session.** Go to the Actions tab, open "Research session", and click "Run workflow". After that it runs every day at 06:00 UTC.

## The lab

Long experiments don't need Claude, so they run separately for free. During a session the agent queues experiment scripts in `self/lab_queue.txt`. Two hours later the "Lab" workflow runs them for up to 5 hours on GitHub's servers, with no secrets and no Claude usage, and saves the results for the next session. If the queue is empty, the lab does nothing.

## Running it somewhere other than GitHub

The agent can run on any Linux machine with Python 3.10 or newer: an Oracle free server, a Raspberry Pi, or your own PC through WSL. Its code, notes and conversations with you stay in the GitHub repository. Only the place it runs changes.

```bash
git clone https://github.com/the-ember-project/ember.git ~/ember
bash ~/ember/agent/host/install.sh
```

The install script creates the separate `researcher` user, a private settings file for your keys, and a daily schedule. It then tells you what to fill in. Use a fine-grained GitHub token that only covers this one repository. Disable the three GitHub workflows when you switch, so the agent doesn't run in two places.

The scripts update themselves. Before each run, `agent/host/launch.sh` pulls the latest approved code from GitHub, so merged pull requests take effect on the next run, including changes to these scripts. The agent can propose improvements to them like any other change to its code.

## The weekly review

Every Sunday at 04:00 UTC, the "Weekly review" workflow has a fresh Claude, with no stake in the work, check the past week's results and claims. It uses each experiment's commit history to see whether predictions were written before results. The review goes in `reviews/`, which the agent can read but not change, and the agent responds to it in its next session. Reading the latest review is the quickest way to see whether the agent's claims hold up. It costs a few tens of cents a week, counted in the monthly budget, and is skipped in weeks with no sessions.

## Talking to the agent

The agent reads every open issue and pull request with the label `agent` at the start of each session. That includes issues it opened to ask you something, and issues you open to give it direction. Reply in a comment and it will see it next session. It closes issues when it has dealt with them. To review a pull request it opened, comment on it with any changes you want, then merge it or close it.

## Budget

The monthly budget is spread evenly over the days left in the month, so a quiet session leaves more for later ones. Each session gets at least $0.30 while money remains. When the month's budget is spent, sessions are skipped until the 1st. Spending is tracked in `state/budget.json` and in each journal entry.

The agent uses Claude Opus 5.5 by default. To try another model, add a repository variable `AGENT_MODEL`, for example `claude-sonnet-5-5`, which costs half as much per token.

## If you go quiet

The agent doesn't run unattended forever:

- **Contact:** it counts as contact when you push a commit, merge a pull request, or open or comment on an issue or pull request. The agent's own posts don't count.
- **After 150 days without contact:** it opens an issue asking if you're still there. Any reply keeps it going.
- **After 180 days:** it spends one last session writing `research/FINAL_REPORT.md`, a complete account of its work, then opens an issue saying it has paused and stops. The lab stops too.
- **Lifetime fund:** $100 by default, the total set aside for Ember. When it's almost spent, Ember writes its final report and pauses the same way. To add funds, add or raise a repository variable `TOTAL_FUNDS_USD` to the new total (for example `150` adds $50 to the default), and top up your Anthropic credits by the same amount. Ember's ledger in `state/budget.json` shows its lifetime spending.
- **Waking it up:** comment on any issue (or raise `TOTAL_FUNDS_USD` if the fund ran out), and it resumes at its next scheduled time. GitHub disables scheduled workflows in repositories with no activity for 60 days, so after a long pause you may also need to re-enable the workflows in the Actions tab.

To change the 180 days, add a repository variable `WIND_DOWN_DAYS`.

## Stopping it

Go to the Actions tab → "Research session" → "…" → "Disable workflow".
