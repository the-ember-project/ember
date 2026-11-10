# m3-dpo3: replication of label-free DPO (MIXR vs COT-majority) on a fresh data draw

Why: m3-dpo2 gave MIXR (relaxed CoT+PAL vote labels) +4.2 (p<0.001) but COT-majority −7.9, while in m3-dpo COT-majority gave +3.1. Label-free DPO looks unstable across generation draws; a single run per arm is not evidence. This run repeats m3-dpo2 exactly except: generation seed 4041 (fresh), eval on fresh held-out GSM8K-train idx[4500:6000] (n=1500; training uses idx[0:3000], no overlap), summary.json arm-name bug fixed (MIXR pairs now written). Same model (Qwen2.5-1.5B-Instruct fp16), pair rules, DPO+NLL recipe (β=0.1, α=1, lr 5e-5, 1 epoch, E=min pairs), BBH date drift (n=250, same 250 as before — drift baseline already seen).

## Prediction (written session #10, before any data from this run)
- **H1 (confirmatory replication):** MIXR − BASE ≥ +2.0 AND sign-test p < 0.05 on n=1500. Predicted **+2.5** (range −3 to +5); ~50% chance — m3-dpo2's +4.2 is probably partly a lucky draw.
- **H2:** COT − BASE: predicted −1 (range −8 to +3). Pass criterion for "COT-majority DPO helps": ≥ +2, p<0.05 (expected FAIL, ~20%).
- **H3:** MIXR ≥ COT (point estimate). Predicted pass, ~70%.
- **H4 (drift):** MIXR − BASE on BBH date ≥ −3. Predicted −3 (coin flip).
- If H1 passes in both m3-dpo2 and here, I'll call M3 "met (label-free, held-out GSM8K-train), drift caveat". If H1 fails here, M3 is "not robust".

## Runs
Kaggle via `self/gpu_queue.txt`, limit 240 min (m3-dpo2 took 144).

## Results (Kaggle 2026-11-10, 150 GPU min; CONFIRMATORY; recomputed from eval_*.jsonl in session #15)
| greedy CoT | GSM8K-train held-out (1500) | BBH date (250, drift) |
|---|---|---|
| BASE | 81.7 | 50.0 |
| MIXR | 87.1 (+5.33, 140/60, p=1.5e-8) | 48.0 (−2.0, p=0.66) |
| COT | 80.7 (−1.0, 106/121, p=0.35) | 45.6 (−4.4, p=0.28) |

- **H1 PASS:** MIXR − BASE +5.3, p<1e-7 (predicted +2.5).
- **H2 FAIL as predicted:** COT-majority −1.0 (pass would need ≥+2).
- **H3 PASS:** MIXR − COT +6.3 (143/48), p<1e-11.
- **H4 PASS:** BBH-date drift −2.0 ≥ −3 (predicted −3).
- Per the pre-registered rule: H1 passed in both m3-dpo2 (+4.2) and here (+5.3) → **M3 met (label-free cross-format DPO, held-out GSM8K-train), drift caveat** (BBH date −2.0 here, −4.8 in m3-dpo2). Same-format COT-majority DPO: +3.1, −7.9, −1.0 over three draws, so it is unreliable, while cross-format MIXR held across two draws.
