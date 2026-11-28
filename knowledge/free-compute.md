# Free compute options (thread 4) — checked 2026-11-28, session #49

Binding constraint: Kaggle 24 GPU-h / rolling 7 days (~100 h/month of T4×2 jobs). m3-dpo-iter waited ~3 days for quota.

| Option | Free amount | Headless/API? | Fit |
|---|---|---|---|
| Kaggle (in use) | ~30 h/week T4×2 (we cap 24) | yes (kaggle API) | current |
| **Modal Starter** | $30/month credits, renews monthly (modal.com/pricing, fetched) | yes: Python SDK, `modal run`, token auth (MODAL_TOKEN_ID/SECRET) | best candidate |
| Lightning AI | 15 credits/month (sources disagree: ~22–80 h, maybe one-time); phone verification | studio-based, less suited to CI | weaker |
| Colab free | ~15–30 h/week T4, interactive notebooks | no supported headless API | unsuitable |

Modal prices (fetched 2026-11-28, per second): T4 $0.000164 (~$0.59/h), L4 $0.000222 (~$0.80/h, 24 GB, bf16),
A10 $0.000306 (~$1.10/h); CPU $0.0000131/physical core/s; memory billed too. A 4-core/16 GB L4 job ≈ $1.0–1.1/h →
~25–30 L4-hours/month on $30 (~+25–30% over Kaggle, single GPU per job). Workspace budget = monthly usage cap
(modal.com/docs/guide/budgets), so overspend can be blocked. Card requirement: sources conflict (several say no card
for Starter; blaxel.ai says a payment method is required) — human to check at signup.
Status: asked the human in an issue (session #49).
