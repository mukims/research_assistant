# Hypothesis search — grounded conjectures from the corpus, best-of-N before MCTS

*2026-10-21 · base `ab5931e` (branch `feat/v3-gemma-images`, index v3) · replaces the
Antigravity plan "MCTS Scientific Hypothesis Tree Engine"*

## 1. Where this starts

The Antigravity plan proposed an MCTS engine rewarded by Agent 8's verdict on the
hypothesis, cosine distance from the query, and a model-graded testability score.
An implementation of it sits uncommitted in the working tree
(`research_assistant/brainstorm/`, `ui/mcts_tab.py`, `scripts/mcts_hypothesis.py`,
`tests/test_mcts.py`, `output/mcts_test.md`, a second tab in `app.py`). Checked
against the repo:

| Problem | Evidence |
|---|---|
| Judging the hypothesis rewards restating the literature | The judge asks whether passages support a claim and requires a verbatim span; a new conjecture is in no passage. The draft's smoke run scored groundedness 1.0 because its "primary claim" restated a graphene-nanoribbon result (Li doping ≈1.7 e/nm). Its conjecture, that nanoribbon interlayers suppress Li dendrites, was never checked |
| Distance from the query measures drift, not novelty | `evaluate_novelty` rewards whatever is furthest from the question |
| The model grades its own work | `LLM_MODEL` and the judge resolve to the same model on both backends |
| The "analogy band" never changes | `retrieve_cross_domain_chunks` embeds the root query once and caches it, so every expansion at depth ≥ 2 sees the same 60-nearest chunks; no `EXCLUDED_FROM_EVIDENCE` filter, and 7,531 of the 46,286 chunk vectors in v3 are figure images |
| Errors are rewarded | a judge failure scores 0.5 |
| 15–20 MCTS iterations are best-of-N | UCT needs many visits per node; every rollout is a full generation |
| The example topics are not in the corpus | 702 summaries in v3: 0 mention MXene, dendrites or ferrite; 1 supercapacitor; 6 battery |
| The tab runs inside the session | a refresh kills the run, which is the failure `shared/run_jobs.py` exists to prevent |
| It breaks the suite | `test_the_six_tabs_are_there_in_order` fails |

## 2. Decisions

### A. Premises are judged; the conjecture is not

Every proposal is split into **premises** and a **conjecture**. A premise is one
sentence that one passage states, citing that passage's key (`C7`). The conjecture is
the model's own and carries no key; every surface labels it "not stated by any source".

Each premise is judged the way the citer judges a candidate
(`agent5_batch_citer._cite_by_judge`): the cited chunk plus
`JUDGEMENT_NEIGHBOUR_WINDOW` neighbours (`expand_neighbours`), `assemble_evidence`,
one `judge()` call. The citer's acceptance rule sets the credit:

| verdict (after `enforce_rubric`) | `span_verified` | credit |
|---|---|---|
| Supports | `True` | 1.0 |
| Partially supports | `True` | 0.5 |
| anything else, no verified span, or a judge error | — | 0.0 |

A **Contradicts** verdict on any premise sets the candidate aside. Verdicts are cached
per (claim, chunk) for the run, so a mechanism's premises are judged once however many
candidates inherit them. Agent 5, Agent 8 and `judgement/` are not modified.

### B. Novelty is distance from the nearest paper; relevance is a floor

`novelty` is the conjecture's distance to its nearest summary (`1 − max cos`),
expressed as a percentile of the corpus's own nearest-neighbour distances: "further
from its nearest paper than this share of papers are from theirs". It is relative to
this corpus, not the literature, and the report says so. The nearest paper is shown
beside the score.

`relevance` is cos(query, conjecture), with the query embedded as a query and the
conjecture as a document, the way retrieval compares them. Below
`IDEATION_RELEVANCE_RATIO` (0.9) × the similarity of the weakest paper the gate kept,
the candidate is off topic and set aside.

### C. Testability is five checks in code

The checks are:
- `system` names the system, in at least 3 words.
- `observable` names the observable, in at least 3 words.
- `prediction` gives a direction or a number.
- `null_result` is stated and differs from the prediction.
- `method_key` cites a premise the judge accepted or partially accepted.

`testable` = passed / 5. No model grades it.

### D. Paper-level bands over the summary vectors

The summary vectors (`physics_summaries_v3`, ids `sum::<document>`, 702 today) are
read once per process, and distances are exact numpy. Papers are ranked by distance
to the query. The band is the ranks between `IDEATION_BAND_LO_PCT` (0.10) and
`IDEATION_BAND_HI_PCT` (0.50), minus the papers the gate kept.

The band uses percentiles, not absolute distances, for two reasons:
- Cosine distances are model-specific.
- The corpus is uneven: 143 graphene, 100 LLM and 60 quantum-transport summaries, with stock-market and cover-song papers at the far end.

For each mechanism, the bridge papers are the band papers most similar to the
mechanism's **abstraction**: the mechanism restated with no material, system or field
names. Each paper is used once per run. Its passages come from
`hybrid_search(abstraction, doc_filter={paper}, exclude_types=EXCLUDED_FROM_EVIDENCE)`,
which is the repo's stage-1 → stage-2 pattern. Every `hybrid_search` call in the
package passes `EXCLUDED_FROM_EVIDENCE`.

### E. Coverage is the gate's call

The `DOC_SELECT_K` nearest summaries go through `retrieve.gate_documents`. If it keeps
fewer than `IDEATION_MIN_NEAR_PAPERS` (3), the search is refused before any generation
call. The refusal names the papers it found and points to the Research idea tab. When
the gate rejects everything it returns the top summary alone, so a question the corpus
does not cover always lands here.

### F. Best-of-N, breadth-first, fixed depth

The search runs root → `IDEATION_MECHANISMS` (3) mechanisms from the near passages →
`IDEATION_BRIDGES` (2) bridge papers each → one proposal per (mechanism, bridge).
A proposal is a title, a conjecture, 2–4 premises (at least one from the bridge
paper) and a protocol. Mechanisms carry 2–3 premises.

Proposals are scored per A–C and combined with a preset's weights:

| preset | grounded | novelty | testable |
|---|---|---|---|
| balanced | 0.4 | 0.3 | 0.3 |
| grounded | 0.6 | 0.15 | 0.25 |
| radical | 0.3 | 0.5 | 0.2 |

They are then ranked, with set-aside candidates last and their reason shown. Adaptive
depth is dropped: once premises are what gets judged, more depth adds premises, not
evidence.

Cost for 3 × 2 (an estimate, not measured): 8 model calls (gate, mechanisms, 6
proposals), 18–33 judge calls (~5.4k-token rubric each) and ~10 embeddings.

### G. A background job, one run at a time

The tab starts the search with `run_jobs.start_job(..., origin="ideation")` and
reports through `pipeline_status` under a new stage, `ideate`. A search is refused
while `pipeline_status` shows another run active (the v3 migration, an ingest). This
holds in the tab, the CLI and the eval. Results are written to
`data/ideation/<run_id>.json` and `.md`.

### H. Evaluation before MCTS

The eval uses six questions from clusters the corpus covers
(`eval/cases/ideation_queries.jsonl`; the maintainer edits the list), two runs each,
and three arms:

- **chat**: Agent 7, one turn, `hypotheses` lens. Rated by hand only, since it has no premise structure.
- **sampled**: N = mechanisms × bridges independent proposals from the near passages, scored and ranked as in F. Same budget as best_of_n, no bridges.
- **best_of_n**: the search in F.

Automatic metrics per arm:
- the top candidate's grounded, novelty, testable and composite scores
- premise acceptance across all premises
- the set-aside share, by reason
- calls and seconds
- the run-to-run spread

A blind sheet (arm labels hidden, the key kept in a separate file) is rated 1–5 on
plausible, novel, testable and would-pursue.

**Rule:** bridges are worth keeping if `best_of_n` beats `sampled` on "would pursue"
and on premise acceptance by more than the spread. Phase 2 starts only then.

### I. Phase 2, not planned yet: MCTS as budget allocation

If H passes, the same operators become MCTS actions:
- A mechanism node's next child is its next-ranked bridge (progressive widening).
- A leaf's value is its candidate's Phase 1 composite. There are no rollouts.
- UCT chooses which mechanism gets the next bridge.

It ships only if it beats `best_of_n` at equal model + judge calls by more than the
spread. The draft's UCT and backpropagation code is kept on `wip/mcts-draft` for this.

## 3. Not changed

Agent 5, Agent 8, `judgement/`, `hybrid_search`, `gate_documents`, the other tabs'
behaviour, the dependency set.

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `IDEATION_DIR` (code) | `data/ideation` | results (`<run_id>.json`, `.md`) |
| `CITATION_IDEATION_MECHANISMS` | 3 | mechanisms per search |
| `CITATION_IDEATION_BRIDGES` | 2 | bridge papers per mechanism |
| `CITATION_IDEATION_MIN_NEAR_PAPERS` | 3 | gate survivors needed, else refused |
| `CITATION_IDEATION_NEAR_PASSAGES` | 8 | passages from the near papers |
| `CITATION_IDEATION_BRIDGE_PASSAGES` | 3 | passages per bridge paper |
| `CITATION_IDEATION_PASSAGE_MAX_CHARS` | 7000 | passage text per prompt |
| `CITATION_IDEATION_BAND_LO_PCT` / `_HI_PCT` | 0.10 / 0.50 | the band, as distance-rank percentiles |
| `CITATION_IDEATION_RELEVANCE_RATIO` | 0.9 | × weakest kept paper's similarity = relevance floor |
| `CITATION_IDEATION_MIN_ACCEPTED_PREMISES` | 2 | accepted + partial premises needed |
| `CITATION_IDEATION_TEMPERATURE` | 0.7 | proposals |
| `CITATION_IDEATION_MECHANISM_TEMPERATURE` | 0.3 | mechanisms |
| `CITATION_IDEATION_NUM_CTX` | 8192 | Ollama context for both |
| `IDEATION_PRESETS` (code) | see F | weights |

## 5. Errors

| condition | behaviour |
|---|---|
| gate keeps fewer than 3 | `CorpusCoverageError`, no generation call; the tab shows the papers found |
| another run active | `BusyError`; CLI exit 2; tab warning naming the stage |
| a reply is not the JSON asked for, twice | that mechanism list or proposal is skipped, with a warning |
| a premise cites a key not shown | premise dropped, warning |
| a premise list is over its cap | extra premises dropped, warning |
| the judge raises | premise credit 0, `error` recorded, run continues |
| the band holds fewer papers than asked | fewer candidates, warning |
| cancel pressed | stops before the next proposal; the job shows "stopped" |

## 6. Results

*Filled in after `scripts/evaluate_ideation.py --score --runs 2` on index v3.*
