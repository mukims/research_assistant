# Research Assistant — merged pipeline design

*2026-09-07*

## 1. Purpose

Build one project from two existing repos — `tech_ireland` and `citation_builder`
— that share a lineage but diverged. The merged repo keeps `tech_ireland`'s
infrastructure (which is strictly better in every module the two have in common)
and restores the features `tech_ireland` dropped when it was slimmed down for
deployment.

Target: `/Users/shardul/Downloads/research_assistant`. No git repository is
initialised — that is the user's call to make later. No corpus or model weights
are copied; the pipeline rebuilds its own data.

### What the system does

From a one-line research idea it finds a seed paper, mines that paper's
reference list, downloads the open-access PDFs it can find, and indexes them
two ways — chunk-level into a hybrid vector + keyword store, and paper-level as
a one-paragraph summary. It then answers two questions: *what has already been
done on this idea* (related-work synthesis) and *which source backs this
sentence* (a LaTeX citation).

## 2. Provenance

`tech_ireland` is the base. Every module the two repos share is taken from
`tech_ireland`; `citation_builder` supplies the agents and tests that
`tech_ireland` lacks.

| Component | From | Change on the way in |
|---|---|---|
| `shared/llm.py` | tech_ireland | **+ `chat_stream()`** (new, §5.2) |
| `shared/ingestion.py` | tech_ireland | rewired to `shared/manifest.py` (§5.4) |
| `shared/search.py` | tech_ireland | stale "not yet in this repo" comments removed |
| `shared/retrieve.py` | tech_ireland | as-is |
| `shared/fetch.py` | tech_ireland | as-is |
| `shared/source_key.py` | tech_ireland | as-is |
| `shared/db.py`, `log.py`, `retry.py` | tech_ireland | comment fixes only |
| `shared/manifest.py` | **new** | §5.4 |
| `schemas.py` | **new** | §5.3 |
| `config.py` | tech_ireland | `DATA_DIR` default, env helpers, `DETECTRON_CONFIG` fix |
| `prompts.py` | tech_ireland | superset already; "planned" comments corrected |
| Agent 0 discoverer | tech_ireland | import paths |
| Agent 1 extractor | **both** | GROBID + regex fallback (§5.1) |
| Agent 2 fetcher | tech_ireland | returns `DownloadedPaper` |
| Agent 3 ingestor | tech_ireland | consumes `DownloadedPaper`; legacy shim dropped |
| Agent 4 assistant | tech_ireland | import paths |
| Agent 5 batch citer | citation_builder | ported to `shared.llm` |
| Agent 6 manual ingestor | citation_builder | import paths |
| Agent 7 research chat | citation_builder | ported to `shared.llm.chat_stream` |
| `orchestrate.py` | tech_ireland | GROBID dead-end branch removed (§5.1) |
| `watch.py` | citation_builder `master_orchestrator.py` | renamed, rewired |
| `app.py` | tech_ireland | + 2 tabs, duplicated block extracted |
| `tests/` | citation_builder | ported; + 2 new files |
| CI workflow | citation_builder | paths updated |

### Not carried over

`evaluate_rag.py` and `requirements-eval.txt` (Ragas evaluation — deferred by
request, not rejected); `attic/`; `environment.yml` (`requirements.txt` is the
pinned pip equivalent and `tech_ireland`'s is better constrained); every JSON
state file; both vector stores; `bm25_index.pkl`; the `drafts/` and
`pulled_pdfs/` contents.

`model_final.pth` (~830 MB) is **symlinked** to the existing checkpoint at
`/Users/shardul/Downloads/tech_ireland/model_final.pth`, not copied. It is
optional at runtime: with the file absent, `layoutparser` downloads PubLayNet
weights on first use, and with layout detection off entirely the ingestion path
is text-only.

## 3. Layout

```
research_assistant/
├── pyproject.toml               editable install; package + test config
├── requirements.txt             core (UI, orchestration, text-only ingestion)
├── requirements-layout.txt      optional: detectron2 + torch layout stack
├── requirements-test.txt        the three light packages CI installs
├── Dockerfile  .dockerignore  .gitignore
├── publaynet_config.yaml
├── model_final.pth              → symlink
├── README.md  ARCHITECTURE.md  HOW_TO_USE.md
│
├── app.py                       Streamlit UI (also the container entry point)
├── orchestrate.py               LangGraph — one research idea, end to end
├── watch.py                     watchdog daemon — directory-driven
│
├── research_assistant/          the package
│   ├── config.py                every model name, path and tunable
│   ├── prompts.py               every prompt sent to a model
│   ├── schemas.py               inter-stage contracts (§5.3)
│   ├── agents/
│   │   ├── agent0_discoverer.py      idea → seed paper
│   │   ├── agent1_extractor.py       PDF → reference list
│   │   ├── agent2_fetcher.py         references → open-access PDFs
│   │   ├── agent3_ingestor.py        PDFs → chunks + summaries
│   │   ├── agent4_assistant.py       sentence → citation
│   │   ├── agent5_batch_citer.py     draft → cited draft + report
│   │   ├── agent6_manual_ingestor.py dropped file → corpus
│   │   └── agent7_research_chat.py   multi-turn chat over the corpus
│   └── shared/
│       ├── llm.py          backend-agnostic chat / stream / embeddings
│       ├── ingestion.py    process → upsert → mark → index
│       ├── manifest.py     what has been ingested (§5.4)
│       ├── search.py       hybrid BM25 + dense with RRF
│       ├── retrieve.py     two-stage retrieval + related-work synthesis
│       ├── fetch.py        stream-a-PDF-to-disk-with-validation
│       ├── source_key.py   deterministic document identity
│       └── db.py  log.py  retry.py
│
├── tests/
├── .github/workflows/tests.yml
├── data/                        all runtime state (gitignored, §4.2)
└── docs/superpowers/specs/
```

Three runnable entry points sit at the root because they are the things a user
actually invokes; everything they import lives in the package.

## 4. Structural decisions

### 4.1 Package, not a flat pile

Both source repos put every agent, the config, the prompts, the orchestrators,
the data directories, the JSON state and the 830 MB checkpoint at the top level.
Merged flat, that would be 28 Python files at the root.

The package layout costs one mechanical pass over every `from config import …`
and `from shared.x import …` line. It is paid once, now, and never again.
`pyproject.toml` with `pip install -e .` removes the "only runs from this
directory" fragility both repos have — `config.py` in both anchors paths to
`PROJECT_ROOT` precisely to work around it.

`agentN_*.py` names are **kept**. The numbers imply a pipeline order that is
false for agents 5–7, but every README, architecture doc and diagram in both
repos uses them. Grouping under `agents/` gets the clarity without the churn.

### 4.2 `DATA_DIR` defaults to `./data/`

`tech_ireland` already introduced the indirection but pointed it at
`PROJECT_ROOT`, so by default the vector store, every manifest, the BM25 pickle
and both PDF piles still land in the source tree. Changing the default to
`<project>/data/` separates code from state; `.gitignore` collapses to one
entry. `CITATION_DATA_DIR` still overrides it for read-only or ephemeral hosts.

Under `data/`: `raw/`, `pulled_pdfs/`, `images/`, `logs/`, `physics_vectordb/`,
`drafts/`, `seed_papers.json`, `extracted_citations.json`, `downloaded.json`,
`failed_downloads.json`, `ingested.json`, `bm25_index.pkl`.

## 5. New and changed components

### 5.1 Agent 1 — GROBID primary, regex fallback

The two repos extract references incompatibly, and each approach has the other's
weakness.

- `tech_ireland` posts every PDF to a GROBID server and parses TEI. Output is
  far richer: per-reference title, authors, year, DOI, and a `SequenceMatcher`
  confidence score for each consolidated DOI against the printed reference.
  It needs a Java service. When that service is down, `orchestrate.py` has a
  terminal `GROBID produced nothing → END` edge and the whole run dies.
- `citation_builder` runs `pdftotext` and matches three reference-line regexes
  (`[N]`, `N.`, `(N)`). Weaker output — a raw string per reference, no
  structured fields — but no external dependency at all.

**Merged behaviour.** A single `extract_references(pdf_path) -> list[Reference]`:

1. Probe GROBID once per run. If unreachable, log one warning and use the regex
   path for every PDF.
2. If GROBID is up, batch-process normally. For any individual PDF where GROBID
   returns TEI containing no usable references, fall back to the regex path for
   that PDF alone.
3. Both paths emit `Reference` objects (§5.3). The regex path fills
   `raw_reference` and leaves the structured fields `None`, which Agent 2
   already handles — it asks Crossref to resolve a DOI whenever Agent 1 was not
   confident.

Outcome filing comes from `citation_builder`: a PDF whose references were read
moves to `data/raw/processed/`, one that yielded nothing moves to
`data/raw/failed/`, so a paper is never silently swallowed. GROBID's TEI is
cached under `data/raw/grobid_output/` so a run can be re-parsed without
hitting the server again.

**Consequence for `orchestrate.py`:** the `_after_extract` conditional edge and
its `END` branch are removed. The pipeline degrades to weaker extraction instead
of stopping. The `stopped` state field remains for the no-seed case.

### 5.2 `shared/llm.py` — streaming

Agent 7 streams token-by-token via `ollama.chat(..., stream=True)` directly.
Porting it as-is would mean the merged repo has a backend abstraction that one
of its agents bypasses, so the abstraction gains streaming:

```python
def chat_stream(messages, model=None, options=None) -> Iterator[str]:
    """Yield content deltas. Same backend selection as chat()."""
```

- `ollama`: iterate the streaming response, yield `chunk.message.content`,
  tolerating the dict form the way `_ollama_chat` already does.
- `openai`: `stream=True`, yield `chunk.choices[0].delta.content`, skipping the
  `None` deltas the API emits for role-only and terminal frames.

`options` carries `CHAT_OLLAMA_OPTIONS` (flash attention, quantised KV cache)
for the ollama backend and is ignored by the openai backend, which has no
equivalent.

Agents 5 and 7 and `evaluate_rag`'s eventual return all move onto `shared.llm`;
after this pass no module outside `shared/llm.py` imports `ollama` or `openai`.

### 5.3 `schemas.py` — inter-stage contracts

The agents talk through untyped JSON, and it has already caused a silent
failure in this codebase: `agent3_ingestor._pdfs_from_manifest` exists solely to
defend against two incompatible `downloaded.json` shapes, one of which "made
every entry look like a missing file and ingested nothing at all".

Three frozen dataclasses, each with `to_dict()` / `from_dict()`:

```python
@dataclass(frozen=True)
class Reference:          # agent 1 → agent 2
    raw_reference: str
    source_document: str          # which PDF cited it
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    doi_confidence: str | None = None    # high | medium | low | unknown
    arxiv_id: str | None = None
    pmid: str | None = None

@dataclass(frozen=True)
class DownloadedPaper:    # agent 2 → agent 3
    key: str                      # source_key, the primary identity
    path: str
    provider: str                 # unpaywall | europepmc | arxiv | crossref
    fetched_at: str
    title: str | None = None
    raw_reference: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None

@dataclass(frozen=True)
class SeedPaper:          # agent 0 → orchestrator
    key: str
    path: str
    fetched_at: str
    source: str                   # "search" | "manual-url"
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
```

`from_dict()` ignores unknown keys and defaults missing optional ones, so a
manifest written by an older run still loads. A missing **required** field
raises rather than producing a half-built record — that is the failure mode
being designed out. Agent 3's legacy-shape shim is deleted.

These describe the JSON manifests only. ChromaDB metadata stays as it is:
Chroma requires flat scalar values, and the existing `citation_source` /
`document` / `page` / `type` / `extra_*` convention is load-bearing for both
retrieval stages.

### 5.4 `shared/manifest.py` — what has been ingested

Currently "already ingested?" has two sources of truth that
`get_ingested_documents()` unions on every call:

1. `ingestion_manifest.txt` — a newline-delimited text file **inside the
   ChromaDB directory**, appended to per PDF, with no deduplication, growing
   without bound.
2. A full paginated scan of every metadata record in the collection, 5000 rows
   at a time, run on every ingest.

Replacement: `data/ingested.json`, a JSON object mapping `pdf_key` → ingest
timestamp, written atomically (temp file + `os.replace`, the pattern
`shared/fetch.py` already uses for downloads).

```python
def load() -> dict[str, str]
def contains(pdf_key: str) -> bool
def add(pdf_key: str) -> None
def rebuild_from_collection() -> int      # repair path
```

The collection scan becomes `rebuild_from_collection()` — a repair function run
on demand when the manifest is lost, rather than on every ingest. This is
correctness *and* speed: the current code re-reads the entire corpus's metadata
before ingesting a single file.

Free to do now because the corpus starts empty; later it needs a migration.

### 5.5 `config.py`

- `DATA_DIR` default → `<project>/data/` (§4.2).
- `_env_bool(name, default)` / `_env_int(name, default)` helpers. The literal
  `.lower() in ("1", "true", "yes")` comparison is currently written three
  times and its inverse (`not in ("0", "false", "no")`) once more.
- **Bug fix.** `tech_ireland` currently has:

  ```python
  DETECTRON_CONFIG = os.environ.get(
      "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config", "publaynet_config.yaml")
  ```

  The arguments are inverted — the model-zoo URL is being used as the *environment
  variable name*, so the value is always `"publaynet_config.yaml"` and the lp://
  path is unreachable. Becomes:

  ```python
  DETECTRON_CONFIG = os.environ.get(
      "CITATION_DETECTRON_CONFIG", "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config")
  ```

  With the local `publaynet_config.yaml` still selectable by env var.

### 5.6 `app.py` — two new tabs

Existing: *Research a topic* (LangGraph run), *Cite a draft* (Agent 4),
*How to use* (renders `HOW_TO_USE.md`).

Added:

- **Cite a draft (batch)** — upload or paste a draft, run Agent 5, show the
  cited text, the citation-key mapping, and render the generated markdown
  report inline. Offers the three output files for download.
- **Research chat** — `st.chat_message` / `st.chat_input` over Agent 7, with
  `st.write_stream` bound to `chat_stream()`. Session state holds the
  `ResearchChat` instance so history survives reruns. The `/sources` and
  `/export` commands become UI affordances (a sources expander under each
  answer, an export button) rather than typed commands; the CLI keeps both.

The retrieved-context expander block, currently duplicated verbatim in
`_render_suggestion` and `_render_build`, is extracted to one function.

### 5.7 `watch.py`

`citation_builder`'s `master_orchestrator.py`, renamed and rewired. Watches
three directories with debounce timers and per-directory cooldowns:

| Directory | Action |
|---|---|
| `data/raw/` | Agent 1 → Agent 2 → Agent 3 |
| `data/pulled_pdfs/` | batch ingest via `shared.ingestion` |
| `data/drafts/` | Agent 5 |

It keeps its startup `sync_database()` pass. This is a genuinely different mode
from `orchestrate.py` — reactive and directory-driven versus a single
query-driven run — so both are kept as separate entry points over the same
stage functions.

## 6. Data flow

```
research idea
     │
     ▼
Agent 0  ──► data/raw/<source_key>.pdf        + seed_papers.json  [SeedPaper]
     │
     ▼
Agent 1  ──► GROBID ──(down / empty)──► pdftotext + regex
     │       extracted_citations.json                            [Reference]
     │       raw/processed/ | raw/failed/
     ▼
Agent 2  ──► Crossref → Unpaywall → Europe PMC → arXiv
     │       data/pulled_pdfs/  + downloaded.json                [DownloadedPaper]
     │                          + failed_downloads.json
     ▼
Agent 3  ──► semantic chunks ──► physics_papers    (chunk index)
     │       one summary/paper ─► physics_summaries (stage-1 index)
     │       BM25 rebuild        + ingested.json
     ▼
retrieval: summaries → LLM relevance gate → shortlist → hybrid chunk search
     │
     ├──► Agent 4  one sentence  → \cite{key} + justification
     ├──► Agent 5  whole draft   → cited draft + mapping + report
     └──► Agent 7  conversation  → grounded multi-turn answers
```

Every stage keeps its own on-disk state, so a run that dies partway resumes by
running it again — finished stages no-op.

## 7. Error handling

Unchanged in character from `tech_ireland`, which is deliberate about it:

- **Network calls** return `(ok, reason)` rather than raising, so a caller
  trying several providers moves to the next instead of aborting
  (`shared/fetch.py`, Agent 0's provider walk, Agent 2's four-source ladder).
- **A PDF that is not a PDF** — an HTML paywall page served with HTTP 200 — is
  the most common failure in this pipeline and is caught by magic-byte check
  before anything is written; downloads land in `.part` and are renamed only
  when complete.
- **LLM calls** are wrapped in `@retry` with exponential backoff.
- **The relevance gate fails open**: a gate call that errors keeps the document
  rather than dropping it, and a gate that rejects everything falls back to the
  top summary.
- **Agent 5 aborts rather than guesses.** If the batched citation-need check
  returns a verdict list that cannot be aligned to its input sentences, it
  raises after retries and writes nothing — a misaligned list would attribute
  one sentence's decision to another. This behaviour is preserved exactly.
- **New:** GROBID being unreachable is now a degradation, not a stop (§5.1).

## 8. Testing

Ported from `citation_builder`, which CI runs on Python 3.10 and 3.12 with only
three light packages installed. The heavy stack (ChromaDB, torch, detectron2,
LangChain embeddings) is imported *inside* the functions that use it — load-
bearing, not stylistic, and preserved.

| File | Covers |
|---|---|
| `test_extractor.py` | Agent 1 outcome filing; unique destination naming. **Extended** with GROBID-down → regex fallback |
| `test_fetcher.py` | `paper_filename` identity — same DOI ⇒ same name regardless of queue position or capitalisation |
| `test_ingestion.py` | check/mark bookkeeping; `pdf_key` normalisation |
| `test_search.py` | partition-based top-k equals true top-k; out-of-range dense hits dropped |
| `test_batch_citer.py` | `_cite_keys` parsing; verdict alignment. **Mock retargeted** from `ollama.chat` to `shared.llm.chat` |
| `test_schemas.py` | **new** — round-trip, unknown keys ignored, missing required field raises |
| `test_manifest.py` | **new** — atomic write, dedup, `contains` after `add`, corrupt-file recovery |

CI also keeps the import sweep — every module imported in a bare environment,
catching a broken import no test happens to cover.

**Verification for this build:** the suite plus the import sweep run without
Ollama, GROBID or network access, so the merge can be confirmed sound
mechanically. A live end-to-end run additionally needs a running Ollama daemon
and is out of scope unless requested.

## 9. Out of scope

Deferred by request: `evaluate_rag.py` and the Ragas harness. The prompts it
shares with Agent 4 stay in `prompts.py`, so returning to it later is a
drop-in.

Considered and rejected: renaming `agentN_*.py` to semantic names (churn across
every doc for marginal gain); splitting `shared/` into infrastructure and
domain packages; an async rewrite; abstracting the vector store behind an
interface. None solve a problem this project has.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Import rewrite across ~28 files misses a reference | CI import sweep catches any module that fails to import; run it before declaring done |
| Regex fallback produces lower-quality references than users expect | Log clearly which path each PDF took; record it in `extracted_citations.json` so the provenance is visible downstream |
| `chat_stream` behaves differently across backends | Both implementations yield plain `str` deltas; agent code sees one contract. Ollama-specific `options` ignored by the openai path |
| Streamlit chat state lost on rerun | `ResearchChat` instance held in `st.session_state`, as the existing tabs already do for their results |
