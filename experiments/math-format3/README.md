# math-format3: does a third answer format add to the CoT+PAL vote? (MATH, Qwen2.5-1.5B Q4, CPU lab)

**Question.** G3 (greedy CoT + 3 PAL samples) gains +6.4 on MATH (math-g3). Is that gain about *format diversity*
(decorrelated errors across formats) or just more votes? Test at equal voter count (4 voters) on the same 250
questions as math-g3 (positions 2000–2249), reusing its stored greedy CoT and PAL samples:
- **G3** = vote(CoT, PAL1, PAL2, PAL3) (stored)
- **A_SYM** = vote(CoT, PAL1, PAL2, SYM): swap one imperative program for a *declarative SymPy* program
  (declare unknowns, write conditions as equations, let sympy solve) — a third format.
- **A_CS** = vote(CoT, PAL1, PAL2, CS): swap one program for a sampled CoT (T=0.7) — same format as the anchor.
Same vote rule as math_g3.py (plurality, ties to greedy CoT, else first tied in order). New samples use a fresh seed
(7373), T=0.7, max 1024 tokens. This is the equal-count control the 11-08 review asked for ("mixing vs sampling").

Harness note (set after the 2-question smoke test, before the real run): if a SYM program has no `print(`, the
harness appends `print(answer)` (the smoke showed notebook-style endings). Stored PAL samples did not get this.

## Prediction (written session #20, before any run)
- **Validity:** SYM executes to a number on ≥60% of questions; CS has a \boxed answer on ≥80%; G3 recomputed from
  stored samples reproduces stored g3_ok on 250/250. Otherwise the verdict is marked compromised.
- **H1 (confirmatory):** A_SYM − G3 ≥ +2.0 points AND sign-test p < 0.05 (n=250). Predicted **+0.4** (range −2 to +3);
  expect FAIL — a 1.5B model's declarative sympy is probably weaker than its imperative code, and errors of two
  program styles are likely correlated.
- **H2 (confirmatory):** G3 − A_CS ≥ +1.5 points AND p < 0.05 (a cross-format voter beats a same-format voter at
  equal count). Predicted **+1.5**, p ≈ 0.2; expect to meet the point threshold but fail on p (few discordant pairs).
- Exploratory (reported, no pass/fail): single-sample accuracy of SYM and CS; G3+SYM (5 voters) − G3; A_SYM − A_CS;
  same-wrong-answer counts of greedy CoT with PAL1, SYM and CS.
- No verdict before n=250. Seeds/versions: llama-cpp-python 0.3.36, Qwen2.5-1.5B-Instruct q4_k_m, seed 7373.

## Results
**Run complete, n=250 (session #21, from analysis.txt; one lab run, 115 min).** Confirmatory, verdicts per criterion:
- **Validity: FAIL on SYM.** SYM gives a number on 52.8% (<60%); CS boxed 86.8% (pass); stored G3 reproduced 250/250 (pass). Verdicts marked **compromised** (for H1 only the SYM arm is affected; H2 does not use SYM).
- **H1 FAIL:** A_SYM − G3 −1.6 (2 fixed / 6 broken), p=0.29. Predicted +0.4.
- **H2 FAIL:** G3 − A_CS +0.4 (7/6), p=1.0. Predicted +1.5. A sampled CoT is as good a 4th voter as a third PAL sample.
- Exploratory: single-sample accuracy SYM 22.0%, CS 42.0%, PAL1 34.4%, greedy CoT 50.8%. G3+SYM (5 voters) +0.8 over G3 (n.s.).
  CoT and CS share a wrong answer on 18 of 104 both-wrong questions; CoT and PAL1 on 7 of 51; CoT and SYM on 6 of 42.
  Errors across formats are less correlated, but the sampled CoT is more accurate, and the two effects cancel at the 4th-voter margin.
- **Reading:** on MATH at 1.5B, the *marginal* voter's format doesn't matter. A third format (declarative SymPy) is too weak to help. The G3 gain is not shown to require format diversity at the margin. It still needs ≥2 agreeing PAL samples to overturn greedy, and the base 2 PAL voters weren't varied here. This answers the 11-08 review's equal-count question for MATH: no evidence that cross-format beats same-format at equal count.

(pending)
