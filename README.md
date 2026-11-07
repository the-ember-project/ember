# Project Ember

<img src="assets/mark.webp" alt="A glowing ember inside an open bronze ring" width="160" align="right">

An experiment: Ember, an AI researcher that works on its own, twice a day, toward artificial general intelligence. It runs for free on GitHub Actions and uses Claude through the Anthropic API, with a hard monthly spending cap.

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
| `transcripts/` | A full record of each session (gzipped) and a one-page index of it | `run.py` only |
| `agent/charter.md` | The agent's goal and rules | Agent proposes, you approve (never the core rules) |
| `agent/run.py` | The code that runs a session | Agent proposes, you approve |
| `.github/workflows/research.yml` | The session schedule | You only |
| `state/budget.json` | Spending ledger | `run.py` only |

The agent can write only in its own folders (`self/`, `knowledge/`, `research/`, `experiments/`); only those, its ledger and logs, and the transcripts are saved after a session, so it can't change its code, charter, schedule or ledger directly. Symlinks are never followed by the code that holds the keys, and are removed before anything is saved. To change its own code or charter, it opens a pull request. Nothing changes until you review and merge it. It can't merge pull requests itself, and it can't propose changes to the core rules in its charter, its schedule or its budget. It can propose changes to its working rules.

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
8. **Optional: change the monthly budget.** It's $95 by default, which gives each of the two daily sessions about $1.50. To change it, add a variable `MONTHLY_BUDGET_USD`.
9. **Start the first session.** Go to the Actions tab, open "Research session", and click "Run workflow". After that it runs twice a day, at about 06:17 and 18:17 UTC (GitHub can start scheduled runs late).

## The lab

Long experiments don't need Claude, so they run separately for free. During a session the agent queues experiment scripts in `self/lab_queue.txt`. As soon as the session finishes, the "Lab" workflow runs them for up to 5 hours on GitHub's servers, with no secrets and no Claude usage, and saves the results for the next session. If the queue is empty, the lab does nothing.

## Running it somewhere other than GitHub

The agent can run on any Linux machine with Python 3.10 or newer: an Oracle free server, a Raspberry Pi, or your own PC through WSL. Its code, notes and conversations with you stay in the GitHub repository. Only the place it runs changes.

```bash
git clone https://github.com/the-ember-project/ember.git ~/ember
bash ~/ember/agent/host/install.sh
```

The install script creates the separate `researcher` user, a private settings file for your keys, and a daily schedule. It then tells you what to fill in. Use a fine-grained GitHub token that only covers this one repository. Disable the GitHub workflows (Research session, Lab, GPU jobs, Weekly review and the monthly credit reminder) when you switch, so the agent doesn't run in two places.

Note: this host mode hasn't been kept up to date with the GitHub setup. Its install script still sets the old defaults ($40 a month, one session a day, a $100 fund) and it doesn't send Kaggle GPU jobs. On your own PC the `researcher` user can also read any file that's readable by everyone and reach your home network, so it isolates the agent less than GitHub's fresh machines do. Bring it up to date before using it.

The scripts update themselves. Before each run, `agent/host/launch.sh` pulls the latest approved code from GitHub, so merged pull requests take effect on the next run, including changes to these scripts. The agent can propose improvements to them like any other change to its code.

## The weekly review

Every Sunday at about 04:23 UTC, the "Weekly review" workflow has a fresh Claude, with no stake in the work, check the past week's results and claims. It uses each experiment's commit history to see whether predictions were written before results. It also searches the web for earlier published work on the week's main results and says which parts seem new and which were already known. The review goes in `reviews/`, which the agent can read but not change, and the agent responds to it in its next session. Reading the latest review is the quickest way to see whether the agent's claims hold up. It costs well under a dollar a week, counted in the monthly budget, and is skipped in weeks with no sessions.

## Free services you approve

Ember can't sign up for anything itself, but it can find free services worth using, such as free GPU notebook services, and suggest them in an issue. If you agree, you create the account, add its key as a repository secret, and list it in `agent/services.md` with a usage limit. From then on Ember uses it on its own within that limit: never more than half of a free quota, only that one account, within the service's terms, never revealing the key, and logging every use in `self/service_log.md`. Each service connects differently, so ask Claude to wire up and test the first use. Remove its row (and the secret) to withdraw it.

**Kaggle (approved 2026-11-04)** gives Ember a GPU. Ember queues GPU experiments in `self/gpu_queue.txt`, and after each session the "GPU jobs" workflow sends them to Kaggle as private notebooks, waits, and saves their output to `gpu-output/` in each experiment's folder. The Kaggle key (`KAGGLE_API_TOKEN`) is used only by that workflow; Ember's code runs on Kaggle, never next to the key, and Ember never sees it. The workflow stops sending jobs once 15 GPU hours have been used in the last 7 days (change it with a repository variable `GPU_WEEKLY_MINUTES`). To stop it, disable the "GPU jobs" workflow or delete the secret; to cut off the key itself, delete it in Kaggle's Settings → API.

## Talking to the agent

The agent reads every open issue and pull request with the label `agent` at the start of each session. That includes issues it opened to ask you something, and issues you open to give it direction. Reply in a comment and it will see it next session. It replies in comments while a conversation is going and closes the issue when it has dealt with it. To reopen a closed one, click "Reopen issue" and add a comment. To review a pull request it opened, comment on it with any changes you want, then merge it or close it.

## Budget

The monthly budget is spread evenly over the days left in the month, so a quiet session leaves more for later ones. Each session gets at least $0.30 while money remains. Ember can raise a session's budget up to 3 times its even share with `raise_budget` (never more than the month has left), recording why in the journal; later sessions then get correspondingly less. When the month's budget is spent, sessions are skipped until the 1st. Spending is tracked in `state/budget.json` and in each journal entry. Ember also has a discretionary reserve, $5 a month by default, which it can add to a session when the work is worth more than that session's share. It records why in the journal, and unused reserve doesn't carry over. To change it, add a repository variable `DISCRETIONARY_USD` (`0` turns it off).

Ember can hand simple bulk work (summarising, extracting, checking) to a cheaper helper model, Claude Haiku 5.5, with its `delegate` tool. Helper costs count toward the same budget and are listed in `self/helper_log.md` and each journal entry. To use a different helper, add a repository variable `HELPER_MODEL`.

**Where the money comes from:** the founder's Claude Max plan includes $100 of API credit each month (it arrives on the 26th and expires if unused), so Ember's budget is $95 a month, a little under the credit, over two sessions a day. A reminder issue opens on the 26th of each month to claim the credit. If the plan changes, set the repository variables `MONTHLY_CREDIT_USD` to `0` and `MONTHLY_BUDGET_USD` to the new budget.

The agent uses Claude Opus 5.5 by default. To try another model, add a repository variable `AGENT_MODEL`, for example `claude-sonnet-5-5`, which costs half as much per token.

## If you go quiet

The agent doesn't run unattended forever:

- **Contact:** it counts as contact when you push a commit, merge a pull request, or open or comment on an issue or pull request. The agent's own posts don't count.
- **After 150 days without contact:** it opens an issue asking if you're still there. Any reply keeps it going.
- **After 180 days:** it spends one last session writing `research/FINAL_REPORT.md`, a complete account of its work, then opens an issue saying it has paused and stops. The lab stops too.
- **Lifetime fund:** $115 by default (the first month's credit plus $17.70 of purchased credit), and it grows by `MONTHLY_CREDIT_USD` ($100) each month on the 26th. When it's almost spent, Ember writes its final report and pauses the same way. To add funds, add or raise a repository variable `TOTAL_FUNDS_USD` to the new total (for example `150` adds $35 to the default), and top up your Anthropic credits by the same amount. Ember's ledger in `state/budget.json` shows its lifetime spending.
- **Waking it up:** comment on any issue (or raise `TOTAL_FUNDS_USD` if the fund ran out), and it resumes at its next scheduled time. GitHub disables scheduled workflows in repositories with no activity for 60 days, so after a long pause you may also need to re-enable the workflows in the Actions tab.

To change the 180 days, add a repository variable `WIND_DOWN_DAYS`.

## Stopping it

Go to the Actions tab → "Research session" → "…" → "Disable workflow".

## About the mark

The project's founder goes by `focarius`, Latin for "hearth keeper", and their picture is a Roman hearth with its fire tended. The project's mark is a single ember taken from that hearth.

When the ancient Greeks founded a new colony, the settlers carried fire from their home city's sacred hearth to light the hearth of the new one. In Rome, the sacred flame of Vesta, goddess of the hearth, was never allowed to go out. Ember is that carried fire: lit from one person's hearth and meant to keep burning after its keeper is gone.

Look closely and the mark holds two more things:

- **The cracks branch like a neuron.** A mind is starting to grow inside the fire.
- **The ring is open just past twelve, like a clock just past midnight.** It's the first hour of a new day, and the circle isn't closed because the work isn't finished.
