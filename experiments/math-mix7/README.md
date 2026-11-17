# math-mix7: does adding PAL voters on top of sampled CoT help on MATH? (fresh items)

Qwen2.5-1.5B-Instruct Q4_K_M (llama.cpp, CPU lab). MATH numeric test subset (`m3-math/run.py:load_math`),
`random.Random(2029)` shuffle, **fresh positions 2450–2699** (2000–2449 used by math-g3/-3b/-xscale/-sc4/-format3).
Per question: greedy CoT, 3 PAL samples (T=0.7), 3 sampled CoT (T=0.7), max 1024 new tokens, seed 7171.
Arms (plurality vote, ties to the earliest voter, greedy first): COT (greedy), G3 = greedy+3 PAL,
SC4 = greedy+3 CoT, MIX7 = greedy+3 CoT+3 PAL.

Why: math-sc4 (positions 2000–2249) found G3 = SC4 (57.6 each) at equal voter count, and an *exploratory* MIX7 − G3
of +5.2 (p=0.015). This tests MIX7 on fresh items, replicates G3 on MATH, and asks whether PAL voters add
anything on top of 4 CoT voters (MIX7 vs SC4), where at equal count they didn't.

## Prediction (written session #25, before any run)
- **Validity:** greedy boxed rate ≥ 85% and greedy accuracy within 40–62%; n=250. Otherwise verdicts are compromised.
- **H1 (confirmatory):** MIX7 − G3 ≥ +3 points AND sign-test p < 0.05. Predicted **+4, pass (~50%)**.
- **H2 (confirmatory, G3 replication):** G3 − COT ≥ +3 AND p < 0.05. Predicted **+5.5, pass (~70%)**.
- **H3 (confirmatory):** MIX7 − SC4 ≥ +3 AND p < 0.05. Predicted **+3, fail on p (~35% pass)**.
- Exploratory: G3 − SC4 (predicted ≈ 0), SC4 − COT, token cost (predicted G3 ≈ 2.3×, SC4 ≈ 4×, MIX7 ≈ 5.3×).
- Reading: H1 pass + H3 fail = more voters help, format mix not shown to matter; H3 pass = PAL voters add
  information beyond extra CoT samples once there are enough CoT voters.

## Run
CPU lab `run_lab.sh 290` (resumable; ~2 min/q on the lab machine → ~2 lab runs). `python analyze.py` for verdicts.

## Results (session #27, 2026-11-17; n=250, two lab runs, complete) — CONFIRMATORY
Validity PASS: greedy boxed 88.0% (≥85), greedy accuracy 51.6 (40–62). PAL exec rate 62.1%.

| arm | acc | tokens vs greedy |
|---|---|---|
| COT (greedy) | 51.6 | 1.00× |
| G3 (greedy + 3 PAL) | 55.2 | 2.21× |
| SC4 (greedy + 3 sampled CoT) | 56.0 | 3.99× |
| MIX7 (greedy + 3 CoT + 3 PAL) | 60.4 | 5.20× |

- **H1 PASS:** MIX7 − G3 = +5.2 (18 fixed / 5 broken), p=0.011 (predicted +4).
- **H2 FAIL on p:** G3 − COT = +3.6 (16/7), p=0.093. Met the +3 threshold but p large; G3 on MATH is +6.4 (math-g3) and +3.6 here.
- **H3 PASS** (predicted likely fail): MIX7 − SC4 = +4.4 (14/3), p=0.013.
- Exploratory: G3 − SC4 −0.8 (13/15, p=0.85) — replicates math-sc4 (equal voters: PAL ≈ sampled CoT, PAL cheaper). SC4 − COT +4.4 (p=0.019). MIX7 − COT +8.8 (27/5, p=1.1e-4).

**Reading.** At 4 voters the format of the 3 extra voters doesn't matter (G3 ≈ SC4). Adding a second group of 3 voters of the
other format adds +4.4 to +5.2 either way. H3 says adding 3 PAL voters to SC4 helps; it does NOT show that mixed formats beat
same-format at equal voter count (no SC7 arm), so "more voters help" and "format diversity helps" are not separated here.
MIX7 costs 5.20× greedy tokens, just above M1's ≤5× budget. Cheapest strong option for MATH 1.5B: MIX7 (+8.8 at 5.2×).
Next test if wanted: SC7 vs MIX7 at equal voters (needs 3 more sampled CoT per item, ~1 lab run).
