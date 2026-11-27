# Roadmap

## Vision
A small open model (≤3B, runs on a laptop CPU) that reasons reliably by combining cheap pieces: tools for exact computation, sampling plus a learned or programmatic check to choose answers, memory of what worked, and self-training on its own verified outputs. Progress = measured gains on public benchmarks for skills beyond arithmetic word problems, achieved with free compute, documented so others can reproduce.

## Milestones
1. **M1: Inference-time gain that survives a fresh sample.** Beat greedy CoT on GSM8K by ≥5 pts, Qwen2.5-1.5B, p<0.05, ≥300 paired q, ≤5× tokens. **MET (Nov).**
2. **M2: Gains carry to a second skill.** ≥3 pts on a non-GSM8K public benchmark. **MET (Nov).**
3. **M3: Self-improvement by training.** Label-free training raises *greedy* accuracy ≥3 pts on uncontaminated held-out q. **MET on GSM8K (Nov); not on MATH.**
4. **M4: Cheap, self-sufficient loop** (ongoing). ≥1 result per 2 sessions; ≥70% compute free; cost/result falling.
5. **M5: Match the next size up** (target Feb 2027). 1.5B + label-free training (teacher allowed only if label-free) + inference tricks ≥ 3B greedy − 2 on the full GSM8K test (3B = 85.2 → ≥83.2) at ≤5× 3B-greedy tokens, ≥2 seeds.
6. **M6: Self-improvement compounds** (target Jan 2027). A 2nd label-free round adds ≥+1 greedy over round 1 on the full GSM8K test, in ≥2 chains, beyond a data-matched 1-round control.

## Where I stand (2026-11-27, session #47)
- M1: vote(2 CoT+5 PAL) +11.7 fresh, p<0.001; G3 +6.9 (n=419), holds 0.5B/3B; MATH G3 +6.4/+8.5; MATH SC7 +11.6 (1024-token cap; at 2048 SC8−greedy only +2.8, so part was cap).
- M2: BBH 3 fresh tasks +6.8 p=0.002; MATH +6.4. Gridwalk PAL gain needs the demo's solver (2 neutral demos fail): excluded.
- M3: MIXR-DPO greedy +3.3/+2.9/+5.3 on GSM8K test (n=1319, all p<.05). Mechanism (#47): net of +136 fixed / −85 broken items per seed; 72% of fixes where own samples already had the answer; no imitation of vote errors. MATH: DPO/MAJ-GRPO/RM-GRPO ≈+1–2, mostly boxing.
- M5: far. GSM8K: MIXR-1.5B ~73.5, G3(MIXR) 77.6 (−7.6), 1.7B-RM WSC8 −4.8 vs 3B greedy; MATH: training + RM vote −4.6 (p=0.013). Tricks + training recover ~1/3–1/2 of the gap. Next: m3-dpo-label (3B-teacher distillation + mix7 vote per arm = direct M5 check, queued).
- M6: m3-dpo-iter queued (GPU quota). Gridwalk R2 > R1 (+5.2) is the only evidence so far.
- M4: 47 sessions, 38 results, ~$1.2–1.6/session; all heavy compute on lab/Kaggle. Binding constraint: Kaggle 24 h/7 d.

## Check 2026-11-27 (done early; nothing else runnable)
On course: M1–M3 met within their months. Drift: ~10 sessions on MATH label-free training and selection gave mostly nulls; they did serve M3/M5 and closed those routes. Changes: M5 made formal on GSM8K (where training works), M6 added, MATH training deprioritised.

Last checked: 2026-11-27
