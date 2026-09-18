# The citer measured, then held to the auditor's standard

Marvin's other half. Agent 8 audits citations with a three-slot rubric, span verification, two
guards and a judge measured at 88% strict. Agent 5 writes them with one loose model call over three
chunks and has never been measured at all. This spec gives the citer a ground truth it gets for free,
then uses the judge as its acceptance test, then fixes its query — in that order, because the first
is what makes the other two claims checkable.

## 1. Purpose

Three things, each shippable on its own:

- **G — measure the citer.** A published paper is a labelled dataset for citation: its author already
  decided which sentence cites which work. Strip the markers, run the citer, compare. No labelling
  session; the seeds are already on disk.
- **A — cite only what the judge would pass.** The citer and the auditor apply different standards to
  the same question, so Marvin flags his own citations. Run each candidate through `judge()` before
  inserting a `\cite{}`; the same test writes and checks.
- **B — the query is not the bare sentence.** "This approach outperforms the baseline" searches for
  those words. The auditor's queries resolve pronouns from the paragraph (`contextualize_citation_
  queries`); the citer's do not.

## 2. Findings that shape the design

1. **The citer is one monolithic loop.** `agent5_batch_citer.run_batch_citer` (lines 283–451) reads
   a file, splits it, batch-checks citation need, and for each sentence inline: `hybrid_search(
   sentence, top_k=3)` over all 17,083 chunks, registers a `cite_N` key per `citation_source`, asks
   the model to rewrite the sentence with `\cite{key}` + a reason, parses `CITED:` / `REASON:`.
   Nothing per-sentence is callable, testable or gateable. Its tests (`tests/test_batch_citer.py`)
   cover the `\cite{}` regex and the YES/NO parser.
2. **The chunk metadata carries both names** — `citation_source` (what the citer keys on) and
   `document` (the PDF basename the audit keys on) — but the citer keeps only the first. Scoring a
   citation against the author's needs the second.
3. **Two of the three seeds are in the chunk index** (`doi_10.1038_nature12952.pdf`,
   `arxiv_2007.12504.pdf`). A citer evaluated on their sentences would retrieve the seed's own text.
   `hybrid_search` has an include-filter (`doc_filter`) and a type exclusion, but no document
   exclusion.
4. **Free gold exists.** `extract_seed_citation_claims` on `arxiv_2108.10114v3` yields 22 sentences
   whose author-cited paper is in the corpus, with clean text (`claim`), a three-sentence window
   (`context`), section, and the resolved reference; the audit manifest match (`_match_downloaded_
   paper`) gives the document basename. Three seeds should give ~60. Uncited sentences from the same
   papers are negatives for citation-need.
5. **The auditor's pieces are reusable as they stand**: `judge()`, `compose_context()`,
   `explain_verdict_steps()`, `assemble_evidence()` + `expand_neighbours()`, and
   `contextualize_citation_queries()` — which already tolerates a claim dict with no `ref`.
6. **Agent 4** (`suggest_citation`) uses a different prompt and returns a suggestion rather than a
   rewritten sentence. A different surface; not touched here.

## 3. Design

### 3.1 The seam — `cite_sentence()`

New function in `agent5_batch_citer.py`; `run_batch_citer` becomes the loop that calls it.

```python
@dataclass
class CiteResult:
    original: str
    cited_text: str                 # == original when nothing was cited
    keys: list[str]                 # cite keys present in cited_text
    candidates: list[dict]          # [{key, citation, document, chunk_index, rrf_score}] in retrieval order
    reasoning: str                  # the model's REASON (legacy path) or the judge's reason (A)
    skip_reason: str | None         # "no relevant context found in database" | "context retrieved but
                                    #  the model did not cite it" | "no candidate passed the judge" | ...
    query: str                      # what was actually searched (B makes this differ from original)
    verdicts: list[dict]            # A only: per candidate, the judge's record

def cite_sentence(sentence, resources, key_registry, *, context=None, query=None,
                  exclude_docs=None, paragraph_id=None) -> CiteResult
```

- `resources` = `(collection, bm25, texts, metadatas)` as today.
- `query`: what to search with; `None` means the sentence itself. B supplies the contextualized
  form; `CiteResult.query` records whichever was used.
- `key_registry` is the caller's `citation_source → cite_N` map, mutated as today so keys stay
  stable across a draft.
- `exclude_docs`: documents never returned. Implemented in `hybrid_search` as a new `exclude_docs`
  parameter applied to both retrievers (sparse: filter the index list; dense: `{"document":
  {"$nin": [...]}}` in the where-clause), next to the existing `exclude_types`.
- Behaviour with `context=None` and the judge off is byte-identical to today; the existing tests
  and two new ones on the seam pin it.

### 3.2 G — gold, metrics, harness

**Package.** `research_assistant/eval/` (new): `__init__.py`, `citer_gold.py`, `citer_metrics.py`,
`cases/`. `pyproject.toml` package-data gains `"research_assistant.eval" = ["cases/*.jsonl"]`.
The judge's eval modules stay where they are; moving them is not this spec.

**Gold record** — one per sentence, `research_assistant/eval/cases/citer_<seed-stem>.jsonl`, tracked:

```json
{"id": "c_2108_0007", "seed": "arxiv_2108.10114v3", "kind": "cited",
 "sentence": "<clean sentence, citation markers removed>",
 "context": "<prev «sentence» next>", "section": "methods",
 "section_heading": "2. Model and Methods", "paragraph_id": "p_12", "sentence_index": 3,
 "author_documents": ["doi_10.1103_physrevb.102.075409.pdf"],
 "author_refs": [{"index": 40, "title": "Disorder information from conductance"}],
 "roles": ["evidential"]}
```

- `kind: "cited"` — every sentence with ≥ 1 author citation whose reference is in the corpus
  (`downloaded` in the audit manifest). `author_documents` lists only the in-corpus ones; references
  not in the corpus are dropped from the record because the citer cannot be expected to find them.
- `kind: "uncited"` — a deterministic sample (seeded RNG) of body sentences carrying no citation
  marker, ≥ 8 words, as many as there are cited records. `author_documents` is `[]`.
- `roles` records the audit's role per author citation so need-recall can be sliced to
  `evidential`; software/pointer sentences stay in the file.
- Built by `citer_gold.build(seed_pdf_path) -> list[dict]` from the seed's TEI (via
  `extract_seed_citation_claims`) and the downloaded manifest (via `_match_downloaded_paper`).

**Scoring** — `citer_metrics.score(results) -> dict`, where each result is `{case, needs_cite: bool,
cite: CiteResult | None, run}`:

| number | definition |
|---|---|
| `need_recall` | cited cases with `needs_cite` True / cited cases |
| `need_specificity` | uncited cases with `needs_cite` False / uncited cases |
| `target_precision` | Σ \|cited_docs ∩ author_docs\| / Σ \|cited_docs\|, micro over cited cases that cited (a cited doc is the `document` of the candidate whose key appears in `cited_text`) |
| `target_recall` | Σ \|cited_docs ∩ author_docs\| / Σ \|author_docs\| |
| `sentence_hit` | cited cases where ∩ is non-empty / cited cases that cited |
| `end_to_end` | cited cases where `needs_cite` and ∩ non-empty / cited cases |
| `declined` | cited cases where `needs_cite` and nothing was cited (with `skip_reason` breakdown) |
| `by_role`, `by_section` | `end_to_end` sliced |
| `stability` | across runs, as the judge harness |

The output states, every time: *author citations are a floor — a correct paper the author did not
cite scores as wrong.*

**Harness** — `scripts/evaluate_citer.py`, mirroring `evaluate_judge.py`:

- `--build <seed.pdf> [...]` writes the gold file(s).
- `--score [--cases ...] [--runs N] [--seed-filter stem]` runs need-check and `cite_sentence` on each
  case with `exclude_docs={that case's seed document}` (resolved from the index by stem; a seed not
  in the index excludes nothing) and the seed's own context as `context`; writes
  `data/eval/citer/results/<stamp>-<judge on|off>-<query raw|ctx>.json`.
- `--compare A B` prints the delta table.
- Need-check is batched per gold file exactly as `run_batch_citer` batches it.

### 3.3 A — the judge as acceptance test

Inside `cite_sentence`, when `CITATION_CITER_JUDGE` (config, `_env_bool`, default **off until
gated**, then on):

1. After retrieval, `expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)`.
2. For each candidate in retrieval order: `evidence = assemble_evidence([hit],
   JUDGEMENT_EVIDENCE_MAX_CHARS)`; `verdict = judge(sentence, evidence, context=compose_context(
   context))`. Record it in `verdicts`.
3. Accept the first candidate with `judgement == "Supports"` and `span_verified is True`. If none,
   accept the first with `judgement == "Partially supports"` and `span_verified is True`, flagged
   `partial: True` in the result and in the report.
4. On acceptance the key is inserted **deterministically** — `\cite{key}` before the terminal
   punctuation, the placement the legacy path already normalises to via
   `_restore_terminal_punctuation` — and `reasoning` is the judge's `reason`. The legacy rewrite
   call is not made: the judge chose, code inserts, nothing to parse.
5. If nothing is accepted: no cite, `skip_reason = "no candidate passed the judge"`, and the
   result keeps the best-ranked verdict so the report can show `explain_verdict_steps()` for it —
   the same block the audit prints.

The report (`_report.md`) gains, per cited sentence, the verified span and the judge's account;
per declined sentence, the closest candidate and why it failed. Cost: ≤ 3 judge calls per sentence
that needs a citation, replacing 1 rewrite call.

**Gate:** G, judge off vs on, 2 runs. Expected: `target_precision` up, `declined` up, `end_to_end`
roughly flat or up. Ships if `target_precision` rises and `end_to_end` does not fall by more than the
noise floor; the on/off numbers go in the commit.

### 3.4 B — contextualized queries

- `run_batch_citer` splits the draft into paragraphs on blank lines, then sentences within each;
  every sentence carries `paragraph_id` and its window (`sentence_context`, as the audit builds
  it).
- When `CITATION_CITER_CONTEXTUALIZE` (config, default **off until gated**, then on): before the
  per-sentence loop, the sentences needing a citation are grouped by paragraph and
  `contextualize_citation_queries([{claim, context, paragraph_id}, ...])` is called once per
  paragraph — the same batching the audit uses. `cite_sentence` receives `query=` and searches
  with it; `CiteResult.query` records it. Off, or on failure, the query is the sentence.
- The G harness passes each case's gold `context` and `paragraph_id`, so the same batching applies.

**Gate:** G, raw vs contextualized, judge in whichever state 3.3 left it. Ships if `sentence_hit`
rises and `target_precision` does not fall by more than the noise floor.

### 3.5 Order

0 seam → 1 G, then baseline numbers (judge off, raw query) → 2 A, gated → 3 B, gated. A before B is
the owner's ordering; they are independent.

### 3.6 Testing

`unittest.TestCase`, model and index stubbed as in `tests/test_batch_citer.py` and
`tests/test_evaluate_judge.py`:

- seam: `cite_sentence` with a stubbed `hybrid_search` and `chat` reproduces today's three outcomes
  (cited / declined / no context) and carries `document` on candidates; `exclude_docs` reaches both
  retrievers (`tests/test_search.py`).
- gold: a small inline TEI + manifest → records with the right `author_documents`; uncited sampling
  is deterministic and excludes the abstract.
- metrics: hand-built results → each number; the floor caveat is in `render`.
- harness: stub the citer, check `exclude_docs` is the seed and the output shape.
- A: stub `judge`; Supports-with-span accepted at rank 2 after a rank-1 Unclear; Partially flagged;
  nothing accepted → no cite, `skip_reason`, best verdict kept; deterministic insertion keeps the
  terminal punctuation.
- B: paragraphs split on blank lines; contextualization called once per paragraph; `query` recorded;
  failure falls back to the sentence.

Live: `--build` on the three seeds, `--score` for the baseline and each gate, numbers in commits.

## 4. Not in scope

- Citing at the claim rather than the sentence, multiple cites per sentence (idea D).
- Source preference — primary over review (E); summary-index retrieval for the citer (F); two-stage
  retrieval for the citer (C). Each is a later gate on G.
- Agent 4 (`suggest_citation`) and Agent 7 (research chat).
- Moving the judge's eval modules into `research_assistant/eval/`.
- Changing what the judge is; V1.6 stands.

## 5. Placement

- `research_assistant/agents/agent5_batch_citer.py` — `CiteResult`, `cite_sentence`, paragraph-aware
  split, the two flags' wiring
- `research_assistant/shared/search.py` — `exclude_docs`
- `research_assistant/config.py` — `CITATION_CITER_JUDGE`, `CITATION_CITER_CONTEXTUALIZE`
- `research_assistant/eval/{__init__,citer_gold,citer_metrics}.py`, `research_assistant/eval/cases/`
- `scripts/evaluate_citer.py`
- `pyproject.toml` — package-data
- `tests/test_batch_citer.py`, `tests/test_search.py`, `tests/test_citer_gold.py`,
  `tests/test_citer_metrics.py`, `tests/test_evaluate_citer.py`
