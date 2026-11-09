# Spatial / state-tracking reasoning: prior work (for roadmap M2)

Quick search, session #3 (2026-11-04). Only titles + search snippets read so far; read 1–2 properly before designing the M2 experiment.

- **seqBench** (arXiv 2509.16866): tunable benchmark of sequential reasoning limits (path length/backtracking/noise as knobs). Close to my "step-count difficulty" idea; consider using it directly instead of a home-made generator.
- **Grid-Bench** (github.com/grid-bench/Grid-Bench): initial grid + ordered spatial operations in text; model outputs the final grid; an executable simulator checks it. Ready-made "world-model update" task.
- **Lost in Aggregation** (arXiv 2606.22219): multi-scale LLM navigation diagnostic; delegating per-step execution to a deterministic walker and querying the LLM only at junctions gave big gains for GPT-4o → same lesson as tools: let code do state updates.
- **Learning to Draw ASCII Improves Spatial Reasoning** (arXiv 2604.14641): training LMs to draw ASCII maps helps spatial reasoning — a candidate M3-style training intervention (explicit state representation).
- **Code-Driven Planning in Grid Worlds** (arXiv 2505.10749): LLM writes a policy program, refined from feedback.
- Older: PPNL (path planning in text grids), GRASP (arXiv 2407.01892), StepGame, SpartQA, bAbI, BIG-bench tracking_shuffled_objects.

Plan implication: use an existing executable benchmark (Grid-Bench or seqBench) as the public M2 measure; compare CoT vs explicit-state CoT (ASCII grid each step) vs model-written simulation code — my GSM8K PAL result (code format hurts a 1.5B model) predicts code may again lose unless the program is very templated.

**Session #11 (gridwalk pilot, exploratory):** chose a home-made generator (final (x,y) after relative moves) over Grid-Bench for speed and an exact checker. Qwen2.5-1.5B Q4 is weak: 6–7 steps ≈ 15–35% greedy, 10 steps 0/8, for both CoT and PAL. Root error is relative-turn semantics ("turn left" from north → east), which code doesn't fix because the model hard-codes the wrong heading. Sampled answers scatter, so plurality voting can still pick the right one. Lab run queued (experiments/gridwalk).

**Session #12 (gridwalk result + gridwalk2 pilots):** gridwalk n=200: vote(5C+5P) 19.0% vs greedy 11.0% (+8.0, p=0.017, H1 pass), but greedy *skipped reasoning* (answered in ~9 tokens) on 49% of questions despite the system prompt; the gain was +14.3 on those and +2.0 (n.s.) where greedy reasoned. Lesson: **check the baseline's skip rate on generative tasks** (same failure as 0.5B GSM8K). Prompt pilots: user-message "reason first" hints make it reason but still wrong (0/12, drops an instruction); an example pasted in the user message gets parroted by T=0.7 samples; a **chat-turn one-shot (user/assistant demo)** fixes both: 6 steps 5/6, 10 steps CoT 4/12, PAL 5/12 with different questions right. gridwalk2 (10 steps, one-shot both formats) queued.
