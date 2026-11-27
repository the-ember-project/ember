# Autoresearch agents (read at the 11-27 roadmap check; helper summary of abstracts, not full text)
- Épi 2610.24944: persistent self-improving research agent; hierarchical memory, versioned branches/backtracking,
  independent review before picking experiments; +5.4% on 30 FrontierOR tasks vs Codex at 64% cost.
- Sibyl-AutoResearch 2605.22343: file-backed harness; bounded trials, positive+negative outcomes routed into later
  planning; a *failure registry* (duplicate results, stale numbers, unsupported statistics) linked to harness fixes.
- CORAL 2604.01658: multi-agent evolution; shared notes/skills repository, periodic "heartbeat" reflection, separate
  evaluator; 3–10× improvement rates vs fixed evolutionary search.
What I already do: playbook = failure registry (GPU-run lessons), weekly external review = independent review.
Cheap gap to adopt: tag each playbook lesson with its failure class and the run that taught it (done implicitly).
