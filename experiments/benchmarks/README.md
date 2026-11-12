# Benchmarks

| Name | Version | What it measures | Why | How |
|---|---|---|---|---|
| GSM8K-100 (public) | v1 | Grade-school multi-step math word problems; exact numeric answer | Well-known public benchmark, comparable with the field; small models make both reasoning and arithmetic errors, so it shows the effect of tools/voting | GSM8K `main` test, shuffle indices with `random.Random(1234)`, questions [0:100]. Extra fresh samples use later offsets ([100:300] etc.). Scorer: number after last `####`, else last number in output; exact match. Script: `experiments/gsm8k-baseline/eval.py` |

Known weaknesses: the fallback "last number" scorer can give credit to outputs without a clean final answer (rare); n=100 gives ±~5 pt standard error.

## Results log
| Date | Model | Method | Benchmark | Score | Notes |
|---|---|---|---|---|---|
| 2026-11-02 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy zero-shot CoT | GSM8K-100 v1 | 68% | 12.4 s/q on CPU; ~1% scorer error found (1 mis-score) |

**Note (session #2):** GSM8K scorer = `####` else last number. Hand-check of 100 greedy outputs: 1 false negative (idx 1117), 0 false credits among the 57 fallback-scored correct answers. Comparisons always use the same scorer on both arms. Paired samples: seed-1234 shuffle; offsets 0–299 have greedy CoT answers.
| 2026-11-04 | same | greedy PAL | GSM8K offsets 0–299 | 61.7% | vs CoT 68.3%, p=0.024 (worse) |
| 2026-11-05 | same | greedy CoT | GSM8K offsets 300–599 | 65.3% | lab |
| 2026-11-05 | same | maj5 CoT | GSM8K 300–599 | 71.0% | +5.7, p=0.024, 4.9× tokens |
| 2026-11-05 | same | maj5 CoT + greedy PAL (w1.5) | GSM8K 300–599 | 74.0% | +8.7, p=0.001, 5.5× |
| 2026-11-05 | same | vote 5 CoT + 5 PAL samples | GSM8K 300–599 | 79.0% | +13.7, p<0.001, 7.7× |
| 2026-11-05 | same | greedy CoT | GSM8K 600–899 | 61.3% | lab |
| 2026-11-05 | same | vote 2 CoT + 5 PAL | GSM8K 600–899 | 73.0% | +11.7, p<0.001, 4.75× (M1 full pass) |
| 2026-11-05 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT / vote 5 CoT+5 PAL | BBH tracking_5 + navigate (public, 500 q) | 39.8% / 49.2% | +9.4, p<0.001; scorer caveat in bbh-mixvote README |
| 2026-11-06 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT / vote 5 CoT+5 PAL | BBH tracking_3 + logical_deduction_5 + date (public, 750 q) | 35.7% / 42.5% | +6.8, p=0.002, fixed parser (bbh-mixvote2) |
| 2026-11-07 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy CoT / vote(greedy CoT + 3 PAL) | GSM8K 900–1318 (419 q) | 66.8% / 73.7% | +6.9, p=0.0001, 2.65× (G3 confirmed) |
| 2026-11-07 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT: base / DPO-MIX (primary, FAIL p) / DPO-COT-majority / DPO-gold | GSM8K test full (1319) | 69.8% / 72.2% / 72.9% / 75.4% | +2.35 p=0.072 / +3.1 p=0.011 / +5.5 p<0.001 (m3-dpo) |
| 2026-11-08 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT: base / DPO-MIXR (primary) / DPO-COT-majority | GSM8K-train held-out idx[3000:4500] (1500) | 80.5% / 84.7% / 72.7% | +4.2 p<0.001 / −7.9 p<0.001; BBH-date drift −4.8 / −8.0 (m3-dpo2) |
| 2026-11-08 | Qwen2.5-0.5B-Instruct Q4_K_M (tutor sys prompt) | greedy CoT / G3 | GSM8K 400–799 (400) | 40.0% / 44.3% | +4.2, p=0.012, 2.48× (g3-scale) |
| 2026-11-08 | Qwen2.5-3B-Instruct Q4_K_M | greedy CoT / G3 | GSM8K 0–299 (300) | 75.7% / 81.3% | +5.7, p=0.0023, 2.72× (g3-scale) |
| 2026-11-09 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy CoT / vote 5 CoT+5 PAL | gridwalk 6-step (own, generative, n=200) | 11.0% / 19.0% | +8.0, p=0.017; but greedy skipped reasoning on 49% — gain +2.0 (n.s.) where it reasoned (gridwalk) |
| 2026-11-10 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy CoT / vote 5C+5P / G3 / vote 5P | gridwalk2 10-step one-shot (own, generative, n=200) | 38.0% / 69.5% / 71.5% / 73.0% | +31.5 / +33.5 p<1e-4; skip rate 0%; gain is the program format (5P ≈ mix, n.s.) (gridwalk2) |
| 2026-11-10 | Qwen2.5-3B-Instruct Q4_K_M | greedy CoT / greedy PAL / G3 | gridwalk2 10-step (own, n=150, idx 200–349) | 41.3% / 96.7% / 95.3% | PAL +55.3, G3 +54.0, p<1e-4; skip 0% (gridwalk2-3b) |
| 2026-11-10 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT: base / DPO-MIXR / DPO-COT-majority | GSM8K-train held-out (1500, fresh draw) | 81.7% / 87.1% / 80.7% | +5.3 p<1e-7 / −1.0; BBH-date drift −2.0 / −4.4 (m3-dpo3; M3 replication pass) |
| 2026-11-10 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT: base / SFT-GOLD / SFT-PALV (label-free) / SFT-COTM | gridwalk 10-step (own, 600 fresh) | 46.3% / 70.2% / 63.2% / 47.3% | +23.8 / +16.8 / +1.0; 14-step transfer +23.0/+16.7/+3.0; GSM8K drift ≥−1.3 (m3-gridwalk) |
| 2026-11-11 | Qwen2.5-1.5B-Instruct fp16 | greedy CoT / G3: base / R1 PALV-SFT / R2 (2 rounds) | gridwalk 10-step (own, 600 fresh, seed 9005) | 47.7/69.8 · 68.8/77.2 · 74.0/80.8 | R2−R1 +5.2 p=3e-6; G3(R2)−G3(base) +11.0 p=2e-8; GSM8K drift +1.3 (m3-gw-iter; more-data confound) |
| 2026-11-12 | Qwen2.5-1.5B-Instruct Q4_K_M | greedy CoT / PAL 1 sample / G3 | MATH test numeric subset (public, n=250, pos 2000–2249) | 50.8% / 34.4% / 57.2% | G3 +6.4 (20/4) p=0.0015 at 2.34× tokens; boxed 88.8%; 7 of 20 fixes rescue CoT truncated at 1024 tok (math-g3) |
