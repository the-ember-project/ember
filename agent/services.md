# Approved services

Outside services the human has signed up for on Ember's behalf. Ember may use each one on its own, without asking each time, within the limit set here and the rules in its charter ("Approved services" under "Your resources").

Only the human changes this file. It's in `agent/`, so Ember can't edit it; it can suggest a service, or a different limit, in an issue.

To add a service:
1. Create the account yourself and read its terms.
2. Add its access key as a repository secret (Settings → Secrets and variables → Actions).
3. Add a row below: what it's for, the secret's name, and the limit.
4. Ask Claude to connect the key to Ember's sessions or lab and test it once, since each service is set up differently.

| Service | What it's for | Secret | Limit | Approved |
|---|---|---|---|---|
| Kaggle (account `focarius`) | GPU experiments: about 16 GB NVIDIA GPU, up to 9 hours a job. Queue them in `self/gpu_queue.txt`; the "GPU jobs" workflow sends them after each session. | `KAGGLE_API_TOKEN` | 24 GPU hours per rolling 7 days (of Kaggle's ~30, leaving a margin), enforced by the workflow. Private notebooks only; research only. | 2026-11-04 |
