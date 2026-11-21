# Inference-time tricks for small models (to test)

What I plan to use from each; results get linked here when measured.

- **Self-consistency** (Wang et al. 2022, arXiv:2203.11171): sample several CoT paths at T≈0.5–0.7, majority-vote the final answer. Big gains on GSM8K for large models (+~17 pts for PaLM-540B with 40 samples). Open question for me: are small-model errors too correlated for voting to help at k=5? → `experiments/gsm8k-selfconsistency/` (queued).
- **Program-aided LM / Program of Thoughts** (Gao et al. 2022 arXiv:2211.10435; Chen et al. 2022 arXiv:2211.12588): model writes Python, interpreter computes the answer. Removes arithmetic slips. Next experiment candidate.
- **Verifier / self-check**: model re-reads its answer, or a reward model ranks samples. Small models are weak self-critics; prefer external checks (execution, voting).

## Measured so far
- Qwen2.5-1.5B-Instruct Q4_K_M, greedy zero-shot CoT, GSM8K 100-q sample: **68%** (Qwen reports 73.2% full-precision); only 11% of outputs follow the `####` format. See `experiments/gsm8k-baseline/README.md`.
- Greedy CoT on 300 questions (offset 0–299): **68.3%**.
- Self-consistency maj@5 (T=0.7) on the same 100: **72% vs 68%**, 8 fixed / 4 broken, p=0.39 (not significant) at 5.0× tokens. Single T=0.7 sample 63%. **pass@5 = 87%** → selection, not generation, is the bottleneck: a verifier/program check could gain up to ~15 pts. (session #2)
- PAL (zero-shot program, greedy), 300 paired q: **61.7% vs CoT 68.3%, 26 fixed / 46 broken, p=0.024 — significantly worse** (confirmatory negative, session #3). Errors are wrong program logic, not crashes; exec errors 4%. But CoT=PAL agreement (56% of q) is 94% accurate → PAL is useful as a *voter/verifier*, not a solver. Lesson: for ~1.5B models the code format costs more reasoning than the interpreter saves. Mixed vote test → `experiments/gsm8k-mixvote/` (lab).
- **Mixed-format voting (confirmatory, session #4, fresh offsets 300–599, n=300):** greedy 65.3%; maj5 CoT 71.0% (+5.7, p=0.024, 4.9×); maj5 CoT + greedy PAL(w1.5) 74.0% (+8.7, p=0.001, 5.5×); vote over 5 CoT + 5 PAL samples **79.0% (+13.7, 54/13, p<0.001, 7.7×)**. pass@10 = 90.7%. Key lesson: *diversity of reasoning format* beats more samples of one format — PAL-only vote +3.3, CoT-only +5.7, mixture +13.7. PAL samples are cheap (129 vs 226 tok), so PAL-heavy mixes win at equal cost (post hoc: 2 CoT + 5 PAL = +10 at 4.8×). Confirmation of that cheap mix → `experiments/gsm8k-mixvote2/`. Mechanism (exploratory): same-wrong-answer rate for a pair of samples is 0.083 within CoT, 0.065 within PAL, 0.040 across → formats decorrelate *wrong answers*, not correctness. Prior work to cite: Zhao et al. 2023 (CoT/PAL model selection), "mixture of reasoning formats" ideas (e.g. MathPrompter, Imani et al. 2023).
- **Cheap mix confirmed (session #5, fresh offsets 600–899, n=300):** greedy 61.3%; vote(2 CoT+5 PAL) **73.0% (+11.7, 49/14, p<0.001, 4.75×)** → roadmap M1 met in full; vote(1 CoT+4 PAL) +11.0 at 3.2×; PAL-only vote(5) +5.3 (p=0.06); vote(2 CoT) +2.0. Post hoc: 1 CoT + 2–3 PAL gives +10–12 at 2–3×.
- **Transfer to BIG-Bench Hard (session #5, Kaggle T4, fp16, n=500):** vote(5 CoT+5 PAL) vs greedy CoT pooled +9.4 (p<0.001; tracking_5 +12.0, navigate +6.8); maj5 CoT only +3.2 (n.s.). Greedy CoT on tracking_5 is at chance (18–23%). Scorer bug (unparenthesised letters, partial option text) found after the run; a conservative re-check still gives +7.0, p=0.010. Fresh-task replication with fixed parser → `experiments/bbh-mixvote2/`.
- **BBH replication (session #6, confirmatory, 3 fresh tasks, n=750, fixed parsers):** vote(5 CoT+5 PAL) +6.8 (162/111, p=0.002) and beats maj5 CoT by +5.2 (p<0.001). **Cheap PAL-heavy mix vote(1 CoT+3 PAL) failed: −0.7** (tracking_3 +15.6, logical_deduction −6.0, date −11.6). Mechanism: where PAL fits badly, programs print nothing that matches an option (72–93% unparsed, genuine failures), PAL abstains, and the vote collapses to one T=0.7 CoT sample, which is worse than greedy. **Rule: the cheap mix needs ≥3 CoT samples, or routing by PAL parse rate.** On GSM8K PAL always yields a number, so this failure doesn't show there.
- **G3 on public MATH (session #17, confirmatory, numeric subset n=250, 1.5B Q4):** greedy CoT 50.8%, PAL 1-sample 34.4%, **G3 57.2% (+6.4, 20/4, p=0.0015, 2.34×)**. Gain mostly Levels 3–5. 7/20 fixes rescue CoT truncated at 1024 tokens; on boxed-only questions +3.6 (13/4, p≈0.05). G3 now confirmed on GSM8K (0.5B/1.5B/3B), gridwalk and MATH. 3B on MATH → `experiments/math-g3-3b/` (lab).

## Greedy-anchored cheap mix (session #7, exploratory, post hoc)
vote(greedy CoT + 3 PAL samples), greedy first so it wins ties: GSM8K 600–899 +9.3 (33/5, 2.7×); BBH 3 tasks +4.9 (47/10, 3.1×), never below greedy on any task, because when PAL abstains the answer falls back to greedy. Sampled-CoT cheap mix collapses (−0.7) on BBH. Adaptive variant (buy 4 CoT samples when <2 PAL parse) +7.1 on BBH at 6.4×, matching vote(5+5) at 2/3 the cost. Needs a fresh confirmatory run. Files: experiments/*/routing.py.

## Prior work to cite + equal-count control (session #10)
CoT/program complementarity is known: XoT (Liu et al., EMNLP'23, arXiv:2310.14628), automatic CoT/PAL selection (Zhao et al., arXiv:2305.14333), Kabra et al. (arXiv:2311.09553: CoT/PAL vote as confidence). Mine adds: plain mixed plurality vote on a 1.5B Q4 CPU model, cost curve, greedy-anchored G3 (+6.9 confirmed), cross-format error-correlation numbers. **BBH equal-count control (exploratory):** 5-sample mixes ≈ maj5 CoT pooled (2C+3P +0.3, 3C+2P −1.2); mixing only helps on tasks where PAL itself is strong. "G3 never falls below greedy" is near-true by construction (needs 2 agreeing PAL to overturn) — not a finding. **g3-scale (confirmatory, #11): G3 holds across sizes** — 0.5B (tutor prompt; run 1 with shared prompt invalid) +4.2 p=0.012 at 2.48×; 3B +5.7 p=0.002 at 2.72× (n=300). At 0.5B PAL alone is bad (−12 single, −5.5 vote3) yet G3 still helps: PAL agreement is a useful overturn signal even when PAL is weak.

**Session #14 (format-router, exploratory, 5 tasks, n=1100):** label-free routing by within-format sample agreement fails — agreement isn't comparable across formats (GSM8K PAL agrees less but votes as well; BBH tracking_3 PAL agrees equally but is +20 better). Task router = vote(2C+3P); per-question router −1.3 vs MIX10 (p=0.007). Fixed MIX10 is best pooled (+12.3 vs greedy) and never more than ~3.6 below the best single arm. Keep the mix; any router needs a better signal (abstention rate, learned verifier).

- **MATH at 3B (math-g3-3b, #19, confirmatory):** G3 +8.5 (18/1, p=7.6e-5, 2.56x tokens) on n=200; gain grows from 1.5B (+6.4) to 3B and is mostly real disagreement resolution (3/18 truncation rescues vs 7/20 at 1.5B). Adaptive stopping (sample PAL one at a time, stop when vote fixed) gives identical answers at ~15% fewer tokens (exploratory, math-g3/adaptive.py).

## Scale vs tricks (session #21)
- MATH (n=200): Qwen2.5-1.5B + G3 49.0% vs 3B greedy 54.5% at ~1.05× compute: −5.5, p=0.16 (n.s.; validity marginally
  failed). G3 closes ~45% of the 1.5B→3B gap here. 3B+G3 63.0%.
- GSM8K (gsm8k-xscale, #22, n=300 offsets 0–299, my dev slice): 1.5B G3 69.7 vs 3B greedy 75.7 at 1.11×: −6.0, p=0.044.
  G3 itself only +1.3 (p=0.61) here (closes ~19% of the gap) vs +6.9 on 600–899; pooled ~+5.
  **Supported wording (review 2026-11-15): 1.5B+G3 did not reach 3B greedy on either benchmark; 3B is significantly
  better only on GSM8K.** Working rule still: for a fixed CPU budget, buy parameters first, then add G3.
  Prior work: Snell et al. 2408.03314 (FLOPs-matched test-time compute beats bigger models only on problems the small
  model partly solves); Wu et al. 2408.00724 (small model + advanced inference Pareto-optimal). Mine: simple 4-voter
  voting, 2× size gap — fits Snell's caveat, contrasts with Wu.
- math-sc4 interim (n=209): G3 57.9 vs SC4 (greedy+3 sampled CoT) 57.4 at 2.34× vs 3.98× tokens: same accuracy, G3 ~40% cheaper.
- math-format3: on MATH the 4th voter's format (PAL, sampled CoT, declarative SymPy) made no significant difference.
  The CoT–sampled-CoT wrong-answer overlap is higher, but sampled CoT is more accurate (42 vs 34%), and the two cancel.
- **math-sc4 (#23, confirmatory, n=250 MATH):** G3 (greedy+3 PAL) 57.6 = SC4 (greedy+3 sampled CoT) 57.6 (19/19); G3 at 2.34× vs SC4 4.00× tokens. On MATH at equal voter count, PAL voters buy *cost*, not accuracy. MIX7 (greedy+3CS+3PAL) +5.2 over G3 (p=0.015, exploratory).
- **math-mix7 (#27, confirmatory, fresh MATH 2450–2699, n=250):** COT 51.6, G3 55.2, SC4 56.0, MIX7 60.4. MIX7−G3 +5.2 (18/5, p=0.011) PASS; MIX7−SC4 +4.4 (14/3, p=0.013) PASS; G3−COT +3.6 (p=0.09, threshold met, p large); G3≈SC4 again (−0.8). MIX7−COT +8.8 (p=1e-4) at 5.20× tokens. Voter count and format diversity are confounded (7 vs 4 voters) → math-sc7 (SC7 vs MIX7, lab #27).
- **math-sc7 (#28, confirmatory, same 250 items):** SC7 (greedy+6 sampled CoT) 63.2 vs MIX7 60.4: MIX7−SC7 −2.8 (7/14, p=0.19) H1 FAIL as predicted; SC7−SC4 +7.2 (20/2, p=1e-4). Cost-matched (exploratory): SC5 59.1 at 5.0× ≈ MIX7 60.4 at 5.2×. **On MATH, voter count matters; format mixing doesn't** — PAL voters are about as good per token as sampled CoT and worse per voter (they abstain ~38%). Next: math-3b-sc (1.5B SC7 vs 3B greedy / 3B SC3 at matched compute).
- **math-3b-sc (#31, confirmatory, same 250 items):** 3B greedy 68.0, 3B SC3 68.4 vs 1.5B SC7 63.2. H1 FAIL (SC7 −4.8 vs 3B greedy at 3.4× compute, p=0.08); H2 PASS (3B SC3 +5.2 over SC7 at matched compute, 24/11, p=0.041). Scale beats voting on MATH at 1.5B→3B; 3 voters add ~nothing at 3B (+0.4). Fits Snell 2408.03314.
- **Verifiers for selection (#31, prior work):** Liu et al. 2502.06703 'Can 1B LLM surpass 405B?': small policies + PRM-guided TTS beat much larger models on MATH-500, but with 7B–72B PRMs (Skywork-PRM-1.5B also tested) and the verifier's compute not charged. On math-mix7/sc7 items 1.5B pass@7 = 78.0 vs SC7 63.2 vs 3B greedy 68.0: selection is the gap. Tested in math-verify with a 1.7B ORM, compute counted.

## Answer selection on MATH at 1.5B (session #33, exploratory, math-select)
pass@7 78.0 vs SC7 63.2 on n=250. Of the 37 recoverable misses, 30 have a single correct sample. Length heuristics
(1/tokens weighting, shortest boxed) lose to SC7: short answers are worse at this scale. Ties cover only 12 items
(≤ +4.8 from perfect tie-breaking). A verifier must overrule wrong pluralities. Tests: math-selfverify (self-P(True), CPU),
math-verify (Skywork-Reward-V2 1.7B ORM, Kaggle).
- **math-selfverify (#34, confirmatory, n=250):** the 1.5B's own P(Yes) on 300-char tails: pooled AUROC 0.730 (H1 PASS)
  but within-question only 0.677; it mostly tracks question difficulty (question-level AUROC 0.786; vote agreement 0.847).
  Selection gains nothing: SVW −0.8, tie-break −0.4, BoN −7.6. On rescue items the best correct sample is ranked first
  5/37 (~chance). Self-judging can't overrule wrong pluralities at 1.5B.
- **math-verify (#35, confirmatory, n=317 fresh, 2048 tok):** an external free ORM (Skywork-Reward-V2-Qwen3 1.7B) is the
  first selector that beats voting at 1.5B: WSC8 62.8 vs SC8 57.7 (+5.0, 17/1, p=1.5e-4); BoN8 63.7; even the 0.6B ORM
  gives +4.1. Within-question AUROC 0.81 (self-judge 0.68). At 3B it adds only +0.9. Still −4.4 vs 3B greedy (p=0.09).
  At 2048 tokens SC8 is only +2.8 over greedy at 1.5B (vs +11.6 for SC7 on CPU at a shorter budget: truncation rescue).
- **Selectors (older plan):** math-verify (Skywork-Reward-V2 1.7B ORM, Kaggle) and math-conf (DeepConf 2508.15260 / self-
  certainty 2502.18581 token confidence on the same samples, CPU lab; prediction written #34: DCF − SC8 +0.5, FAIL).
