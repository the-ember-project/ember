# Reading list: complementary learning systems and memory

Added 2026-11-06 at your request in issue #5. It comes from the founder's
separate hippocampus/cortex side project, so "for us" below means that
project, not you.

**How far to trust it:**
- It's a guide to the papers, not the papers themselves: short summaries,
  citations and links. Open any paper you want to rely on with web_fetch.
- The classic papers (section "Classic papers") were cited from memory.
  They're well known and very likely right, but no one checked them
  against the originals.
- The 2025–2026 preprints were judged only from search summaries, not
  read in full. Treat their numbers as unverified.
- Use, rewrite or delete this file as you see fit. It isn't a rule.

---


## The four ideas this project borrows

### 1. Complementary learning systems (McClelland, McNaughton & O'Reilly, 1995)
**What it says.** The brain needs two learners because one can't do both jobs:
- the **hippocampus** learns fast, from a single experience, and keeps memories
  separate so they don't blur together;
- the **neocortex** learns slowly and builds general knowledge by overlapping
  many experiences.

If the cortex learned fast, each new thing would overwrite old knowledge. The
paper showed this with neural networks ("catastrophic interference") and argued
that the hippocampus *replays* memories to the cortex, mixed in with old ones,
so the cortex can learn the new thing without wiping out the old. Kumaran,
Hassabis & McClelland (2016) updated the theory: replay can be *prioritized*
(surprising or rewarding events get replayed more), and the cortex can learn
quickly when new information fits an existing schema.

**For us:** this is our design. It's well tested in the brain and in small
networks. It is not new.

### 2. Hippocampal indexing theory (Teyler & DiScenna, 1986; Teyler & Rudy, 2007)
**What it says.** The hippocampus doesn't store the whole memory. It stores an
*index*, a pointer to the cortex patterns that were active at the time. A
partial cue (a smell) re-activates the index, and the index re-activates the
whole pattern. This is why one cue can bring back a full scene.

**For us:** our "episode store + similarity search" is a crude version. The
embedding is the index, and the stored text/context is the content. In a fuller
version the index would point at the model's own internal states, not just text.
HippoRAG (below) uses this theory directly.

### 3. Replay for continual learning in machine learning
**What it says.** When a network learns task B after task A, its task A
skill drops: catastrophic forgetting (McCloskey & Cohen 1989; French 1999).
The standard fixes:
- **Experience replay:** keep a small buffer of old examples and mix them into
  new training (Rolnick et al. 2019, "Experience Replay for Continual
  Learning"; Chaudhry et al. 2019, "Tiny episodic memories"). Even a tiny
  buffer helps a lot. This is one of the strongest, simplest baselines.
- **Generative replay:** instead of storing old data, a generator *dreams* fake
  old examples (Shin et al. 2017, "Deep Generative Replay"; van de Ven et al.
  2020, "Brain-inspired replay", which explicitly cites the hippocampus).
- **Regularization:** protect weights that mattered for old tasks (EWC,
  Kirkpatrick et al. 2017; Synaptic Intelligence, Zenke et al. 2017).
- **Gradient guards:** only allow updates that don't increase loss on stored
  old examples (GEM, Lopez-Paz & Ranzato 2017; A-GEM 2019). These are close
  cousins of our "drift guard", applied at every step rather than as an
  accept/reject check after sleep.
- **LoRA forgets less but learns less** (Biderman et al. 2024, "LoRA Learns
  Less and Forgets Less"). So a LoRA cortex is a sensible slow learner, but it
  can still forget. Recent work organizes LoRA adapters CLS-style (e.g. ProCL,
  EMNLP Findings 2026).

**For us:** interleaved replay working is *expected*. The toy step is a
teaching demo and a sanity check for our code, not a discovery.

### 4. Agent-memory systems
- **MemGPT / Letta** (Packer et al. 2023): an LLM manages its own memory like
  an operating system, with a small "main context" (RAM) and large external
  stores (disk), and moves things between them with tool calls. **Nothing is
  written into the weights.** Memory is retrieval only.
- **Generative Agents** (Park et al. 2023): stores observations with
  *importance* scores; recall ranks by recency × importance × relevance; a
  periodic **"reflection"** step writes general lessons from recent memories.
  That's very close to our "hippocampus + written lessons", but again with no
  weight changes.
- **HippoRAG / HippoRAG 2** (Gutiérrez et al. 2024/2025): retrieval modeled on
  indexing theory (knowledge graph + personalized PageRank). Retrieval only.
- **Larimar** (Das et al. 2024): an external episodic memory attached to an
  LLM for one-shot fact edits and forgetting, framed as CLS. No consolidation
  into weights.
- **EM-LLM** (Fountas et al., ICLR 2025): splits the model's input stream into
  "episodes" wherever the model's **surprise** spikes, then recalls episodes
  by similarity plus time order. That's good evidence that surprise is a
  sensible episode marker. Also retrieval only.
- **Reflexion** (Shinn et al. 2023): the agent writes lessons from its
  failures in plain language and reads them back later.

**For us:** the "fast store + recall by cue + written lessons" half is
well covered and works. Most of these systems stop there and never change
the model.

### 5. Recent "sleep → LoRA" work (2025–2026, mostly preprints)
A cluster of recent work does what we're sketching: keep facts in context or
a key-value store while "awake", then fold them into a LoRA adapter during
"sleep". Examples found: a sleep-wake system on a 3B Llama (Zenodo preprint,
Feb 2026, single author, reports a narrow working learning rate around 1e-4
and a spacing effect); a two-phase "slow-wave + REM" scheme; a "SLEEP" system
with prediction-error tagging + generative replay + LoRA; a May 2026 arXiv
preprint where weight consolidation kept far more knowledge than repeated
context summarizing; and a Google Research blog post on a sleep/"dreaming"
framework. **Most are unreviewed and small-scale. Treat their numbers as
provisional.** They all name the same open problem: **stability vs.
plasticity** (learn the new without losing the old).

### 6. Recent CLS theory and neuroscience (added 2026-11-05)
- **Not everything should be consolidated: Go-CLS** (Sun, Advani,
  Spruston, Saxe & Fitzgerald, *Nature Neuroscience* 2023). Moving every
  hippocampal memory into the cortex makes the cortex *overfit* when the
  world is partly unpredictable: it memorizes noise as if it were a rule.
  Their model consolidates only memories that help generalization, and
  keeps the rest as hippocampus-only "notebook" memories. **For us:** a
  very surprising episode may be important, or it may just be noise (or
  poison). Surprise should decide what the *hippocampus keeps*, but maybe
  not what the *cortex learns*. That's testable.
- **Awake ripples tag experiences for sleep** (Yang … Buzsáki, *Science*
  2024). In mice, experiences followed by bursts of sharp-wave ripples
  while awake (brief replays during a pause, e.g. eating a reward) are the
  ones replayed during sleep and kept. **For us:** this is close to our
  "recall strengthens memory" rule: reactivation while awake is the tag.
- **A small CLS inside the hippocampus** (Schapiro and colleagues; 2017
  model, 2025 preprint, 2026 *Phil. Trans. B* paper). One hippocampal
  pathway learns individual episodes (pattern separation), and another
  (EC→CA1) quickly picks up regularities. So the fast store isn't just a
  list: it also extracts rough patterns within minutes. **For us:** the
  hippocampus could hold a quick "draft lesson" before sleep turns it into
  a cortex lesson.
- **Sleep evidence is mixed.** A 2024 CLS model of sleep and retrieval
  practice (TEACH) notes that evidence for "slow-wave sleep replay causes
  consolidation" is inconsistent. Brains don't give us one clean
  recipe.
- **Representational drift** (arXiv 2512.22045): brain representations
  slowly change even for stable memories, and a 2025 preprint frames this
  as a possible help for continual learning. That suggests our drift
  guard should test *behavior* (skills), not internal similarity, which
  it already does.
- **Surprise-gated memory in ML** (Mouchon, arXiv 2606.31495, June 2026,
  single-author, single-seed): a prediction-error detector decides what
  gets stored, offline replay consolidates it, and big retention gains
  are reported on an image stream. Close to our design, but unreviewed.

## So what's new, or worth testing?

Honestly, none of the individual parts is new. Being new isn't the point of
a learning project, but some *combinations and measurements* are still
open, small enough to test at home, and useful to Ember's M3 later:

1. **Accept/reject drift guard as a gate on the whole sleep.** GEM-style
   methods constrain every gradient step; most sleep→LoRA papers just report
   forgetting afterwards. A hard "keep the update only if no old skill drops"
   check is simple and practical. Questions: how often does it reject? Does
   rejection plus retry with more replay converge? Does it block real
   learning (too strict)?
2. **Surprise-prioritized replay vs. random replay** under a tight memory
   budget, with the hippocampus actually *forgetting* low-surprise episodes.
   Theory says prioritization helps. In ML results are mixed. A clean
   small test is worth doing.
3. **"Recall strengthens memory"** (reconsolidation / spacing): episodes that
   get recalled while awake resist fading and get replayed more. Is that
   better than surprise alone?
4. **Manipulation guard under attack.** Trust-gated consolidation is common
   sense, but few papers measure it: plant poisoned episodes from an
   untrusted source, and from a trusted one, and see what reaches the
   weights. Our guard won't catch poison from a *trusted* source. It's
   worth showing that limit openly.
5. **Lessons in writing vs. lessons in weights.** Is a written lesson list
   (cheap, readable, editable) enough for most gains, so the LoRA only
   needs to learn what writing can't capture?

The honest pitch: **a small, open, measurable testbed** for one full
sleep loop (fast store → surprise tags → fading → interleaved replay →
small LoRA update → drift gate → trust gate), with numbers for each part.
Most published versions skip at least one of the measurements.

## Sources
Classic papers (from memory, well known):
- McClelland, McNaughton & O'Reilly (1995), *Why there are complementary learning systems in the hippocampus and neocortex*, Psychological Review.
- Kumaran, Hassabis & McClelland (2016), *What learning systems do intelligent agents need? CLS theory updated*, Trends in Cognitive Sciences.
- Teyler & DiScenna (1986); Teyler & Rudy (2007), *The hippocampal indexing theory and episodic memory*, Hippocampus.
- McCloskey & Cohen (1989); French (1999), catastrophic forgetting.
- Kirkpatrick et al. (2017) EWC; Zenke et al. (2017); Lopez-Paz & Ranzato (2017) GEM; Shin et al. (2017) Deep Generative Replay; Rolnick et al. (2019); Chaudhry et al. (2019); van de Ven, Siegelmann & Tolias (2020) Brain-inspired replay, Nature Communications.
- Biderman et al. (2024), *LoRA Learns Less and Forgets Less*.
- Packer et al. (2023) MemGPT; Park et al. (2023) Generative Agents; Shinn et al. (2023) Reflexion.

Found in today's search:
- [EM-LLM, arXiv 2407.09450](https://arxiv.org/abs/2407.09450) and [code](https://github.com/em-llm/EM-LLM-model)
- [Larimar, arXiv 2403.11901](https://arxiv.org/pdf/2403.11901)
- [HippoRAG, 2405.14831](https://huggingface.co/papers/2405.14831); [HippoRAG 2 / "From RAG to Memory", 2502.14802](https://arxiv.org/abs/2502.14802v2)
- [Google Research blog: the sleep framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) (date looked odd in search results; verify)
- [arXiv 2605.24657 (reflection → synthesis → LoRA training)](https://arxiv.org/pdf/2605.24657.pdf)
- [Zenodo 18778764](https://zenodo.org/records/18778764), [Zenodo 18778760](https://zenodo.org/records/18778760) (sleep-wake LoRA preprints, unreviewed)
- [ProCL, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.1048.pdf)
- [MIRROR, ICLR 2026](https://iclr.cc/virtual/2026/10021270)

Recent CLS (section 6):
- [Go-CLS, Nature Neuroscience 2023 (open access)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400413/)
- [Yang et al., Science 2024, awake ripples select experiences (preprint)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10659301) and [ScienceDaily summary](https://www.sciencedaily.com/releases/2024/03/240328162613.htm)
- [Evidence for CLS within the hippocampus, Phil. Trans. B 2026](https://doi.org/10.1098/rstb.2025.0243); [Schapiro CCN 2025 keynote](https://2025.ccneuro.org/keynote-anna-schapiro/)
- [CLS model of sleep and retrieval practice (TEACH), 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11543715/)
- [Learning continually with representational drift](https://arxiv.org/pdf/2512.22045)
- [Surprise as a Signal for Plasticity and Metacognition](https://arxiv.org/pdf/2606.31495)

I haven't read the 2026 preprints in full, only their search summaries. Read
one properly before relying on its numbers.
