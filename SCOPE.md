# Scope

What this repository is, how far each part has got, and where it could go next.
It records a decision to be made, not one already made — see
[Decisions still open](#decisions-still-open).

*Written 2026-10-12 against `main` at `7e69ab2`. Figures come from the
maintainer's machine; see [Where the numbers come from](#where-the-numbers-come-from).*

## What this repo is

One Python package, [`research_assistant/`](research_assistant/), serving three
products that share a core.

**The shared core**

- Ingestion and retrieval: GROBID full text, sentence-window chunking,
  ChromaDB plus BM25, hybrid search with rank fusion, and a two-stage
  paper-summary shortlist ([ARCHITECTURE.md](ARCHITECTURE.md) §4–5).
- A model layer over Ollama, any OpenAI-compatible endpoint, or Gemini
  (`shared/llm.py`).
- The judge: a three-slot claim–evidence rubric whose verdict is derived and
  enforced in code, with a verbatim-span check (`judgement/judge.py`).
- A source assessor and a deterministic reliability policy
  (`judgement/source_assessor.py`, `judgement/policy.py`).
- Run status, telemetry and cooperative cancellation (`shared/pipeline_status.py`).

**The three products**

| Product | What it does | Where it lives | State |
|---|---|---|---|
| **A. Citation verification** | Audits a published paper's in-text citations against the cited papers' text; cites a draft; grades each citation | Tabs *Citation auditor*, *Cite a draft*; Agents 5 and 8; `shared/seed_audit.py`; `judgement/` | Most complete, and the only part with an evaluation harness |
| **B. Literature discovery and synthesis** | Idea → seed paper → reference corpus → related-work synthesis; grounded chat | Tabs *Research idea*, *Research chat*; Agents 0–3, 7; `shared/retrieve.py` | Works end to end; where the project started |
| **C. Paper-to-agent replication planner** | Paper → dependency graph of replication tasks, assigned to agent roles, schedulable and runnable | Tab *Agent task planner*; `research_assistant/tasks/`; `orchestrate_tasks.py` | Newest. Weakest evidence. Largely independent of A and B |

Around them: a LangGraph orchestrator (`orchestrate.py`), a directory watcher
(`watch.py`), a Streamlit UI (`app.py`, laid out in `research_assistant/ui/`), and a
local launcher ([deploy/README.md](deploy/README.md)).

## How well it works

| Measure | Result | Caveat |
|---|---|---|
| Judge, strict accuracy (rubric V1.6) | 88%, against 77% for V1.5 | 32 hand-labelled cases, run twice. Eight labels were corrected partway through, which moved the V1.6 figure |
| Citer: "does this sentence need a citation" | ~55% recall, ~85% specificity | Scored against the seed papers' own citations |
| Citer: picks a reference the authors cited | 18–22% target precision, 12–14% end to end | Author citations are a floor, not the truth: a correct paper the authors did not cite counts as wrong. Compare runs; do not read these as accuracy. See §9.1 of [ARCHITECTURE.md](ARCHITECTURE.md) |
| Reference fetching | 557 downloaded against 890 failed, about 39% | Of the failures, roughly 540 are paywalled, ~120 are open access with no direct PDF URL, ~70 are publisher 403s |
| Corpus (index v2, nomic) | ~44k chunks, 735 paper summaries | One maintainer's corpus. Index v3 (embeddinggemma-2, with figure images) replaces it once the migration has run |
| Task planner | No measurement | Its scorecard rates the share of tasks whose acceptance check passed, not whether the paper's results were reproduced |

The test suite has about 1,600 test functions in 81 files, and CI runs it on
Python 3.10 and 3.12. It was not run when this document was written.

## Known gaps

**The reliability rating rests on weak inputs.**
`assess_source` is given a venue string and little else. Nothing in the package
sets `is_retracted` or `citation_count` from a real source, so the
retracted-source tier fires only if the word "retracted" appears in the title or
venue, and a retracted paper can be rated HIGH. On the draft-verification path
the "venue" is a formatted citation string. The assessor's own rationale text
says its grades are "not verified against Crossref".

**Coverage the design calls for but the code lacks.**
There is no abstract-level fallback when a PDF is paywalled, no second hop
through references, no expert-review queue, no tiering of judge models, and no
detection of uncited claims in an uploaded paper. These are the open items in
[the pitch gap analysis](notimportant/superpowers/specs/2026-09-13-pitch-gap-analysis.md);
the source assessor, reliability policy, judge evaluation set and paywall
upload flow it listed as missing have since been built.

**Discovery quality.**
Several stored seeds have junk titles taken from PDF metadata ("PAPER • OPEN
ACCESS", a bare PII string). The iterative seed search in
`agents/agent0_discoverer.py` is aimed at this; nothing measures whether it
fixed it.

**Domain.**
The prompts and README say physics. Actual use spans nanomaterials, time series,
energy, drug discovery and LLM-for-science. The planner's roles assume
ML-style replication (PyTorch, tensors, training runs), which fits the physics
and materials corpus poorly.

**Operations.**
The tool runs on one machine for one user: `deploy/local.sh` binds the app to
127.0.0.1 and manages GROBID. Access control is an optional `APP_PASSWORD`.
Ingestion runs synchronously under one lock, so the tool is single-user.

**Code health.**
`shared/seed_audit.py` is ~2,000 lines and `agents/agent0_discoverer.py` ~1,800.
`app.py` is a ~90-line page layout; the UI is `research_assistant/ui/`, one
module per tab and per concern, the largest (`ui/audit.py`) ~570 lines.

## Directions

These are not exclusive, but each pulls the repo toward a different centre.

### 1. A citation-integrity checker for scientists

The judge is the spine. The corpus exists to give it something to read.

- Real source metadata from Crossref or OpenAlex: publication type,
  retraction status, citation counts. This makes the reliability rating
  honest.
- An abstract-level fallback for paywalled references, with the rating capped
  below HIGH, plus a guided "upload what you can access, then re-audit" path
  (the upload half exists).
- Flag claims in an uploaded paper that carry no citation (Agent 5's
  need-check already does this for drafts).
- An expert-review queue for Unclear, Low-confidence and Contradicts verdicts.
- Tiered judging: a cheap first pass, a stronger model for hard cases.
- A larger, multi-domain judge evaluation set.

*Why this one:* it has the strongest existing evidence, it is what the README
leads with, and each item above makes an existing claim more true.

### 2. A grounded literature-review and brainstorming assistant

Chat and synthesis are the product; verification is a feature.

- A second hop through references, so a field's foundations are in the corpus.
- A reranker (a cross-encoder model is cached in `data/models`, but no code uses it).
- Deeper per-paper reading than a one-paragraph summary.

*Trade-off:* it is where the project started, but it is the least
differentiated direction and the hardest to evaluate.

### 3. Paper-to-replication workflow

The planner becomes the product.

- Sandboxed execution of generated code, and a scorecard that compares replicated
  numbers with the paper's.
- Role definitions suited to the domains actually used.
- An evaluation: papers with a known replication outcome.

*Trade-off:* a different audience and a different product. `tasks/` reaches
into only a handful of shared modules (`llm`, `log`, `config`, `search`, `db`,
`ingestion`, and one helper from `seed_audit`), so it could be its own package
or repository.

## Suggested direction

1. Take direction 1 as the spine, with direction 2's ingestion and synthesis as
   the way a corpus gets built.
2. Mark the planner experimental in the UI and docs until its scorecard
   measures reproduction, or move it out.
3. Before widening anything: real source metadata for the reliability layer.

## Non-goals, for now

- Multi-user or multi-tenant hosting.
- Papers that are not open access, beyond what a user uploads.
- Running real experiments or closing a lab loop; the pitch's vision slide
  describes this, and nothing in the code starts on it.
- Mixing embedding models within one corpus (documented as unsupported in
  [ARCHITECTURE.md](ARCHITECTURE.md) §6.2).

## Decisions still open

1. Which direction is the spine? Everything above assumes direction 1.
2. Does the planner stay in this repository?
3. Is physics the stated domain, or is the tool domain-neutral? The prompts,
   the README and the default search order currently disagree with actual use.
4. Who is the first user: the maintainer, collaborators, or outside
   researchers? That decides how much deployment and access control to build.

## Where the numbers come from

The evaluation results and corpus counts are from `data/eval/`,
`data/failed_downloads.json`, `data/downloaded.json` and the ChromaDB files on
the maintainer's machine. `data/` is gitignored, so they cannot be checked from
a clone. The harnesses that produced them are in the repository:
[`scripts/evaluate_judge.py`](scripts/evaluate_judge.py),
[`scripts/evaluate_citer.py`](scripts/evaluate_citer.py), and the cases under
`research_assistant/judgement/cases/` and `research_assistant/eval/cases/`.
They need a live model backend, so they are not part of the test suite.
