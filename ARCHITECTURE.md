# Architecture & Design Decisions

Why the Research Assistant is built the way it is. Each section is a decision:
what was chosen, the reasoning, what was rejected, and the trade-off accepted.

---

## 1. Pipeline structure

### 1.1 Separate agents joined by file contracts

The pipeline is agent scripts under `research_assistant/agents/` (`agent0`…
`agent7`) that communicate through files on disk, not function calls:

```
data/raw/*.pdf ─▶ agent1 ─▶ extracted_citations.json ─▶ agent2 ─▶ downloaded.json ─▶ agent3 ─▶ ChromaDB
```

Agents 5–7 (batch citing, manual ingestion, research chat) hang off the same
`shared/` primitives and the same two indexes agent3 builds — they read the
corpus, they don't extend this file chain.

**Why.** Each stage is independently runnable and debuggable — you can re-run
Agent 2 against a hand-edited `extracted_citations.json`, or inspect exactly
what Agent 1 produced. A crash in one stage leaves a clean, resumable artifact.
The stages have very different dependency footprints (GROBID client vs. HTTP
scraping vs. the ML/embedding stack), and file boundaries keep those from
bleeding into each other.

**Rejected.** A single in-memory pipeline object. It would be faster and
simpler to wire, but every run would be all-or-nothing and every stage would
carry every dependency.

**Trade-off.** Serialisation overhead and the need to keep the on-disk formats
stable. In practice the formats change rarely and the readers tolerate legacy
shapes (`agent2` still accepts a flat list of citation strings).

### 1.2 Crash-safe incremental state

`agent2` rewrites `downloaded.json` / `failed_downloads.json` after **every**
paper. `agent3` marks a PDF ingested the moment it is processed, before the
batch finishes.

**Why.** Fetching 60 references takes minutes and hits flaky external servers;
ingestion runs layout detection and embedding per page. A process killed at 80%
must resume at 80%, not 0%. Processing is by far the most expensive step, so a
PDF that is processed but not marked would be re-processed on every subsequent
run forever.

**Trade-off.** More disk writes; a torn write on power loss could corrupt one
JSON file (mitigated by writing whole files, and readers falling back to empty
on `JSONDecodeError`).

### 1.3 One implementation per capability, in `shared/`

PDF processing, chunking, ChromaDB upsert, hybrid search, source-key derivation,
logging, retry — each lives once under `shared/` and is imported by every agent
that needs it.

**Why.** The check/mark bookkeeping in ingestion has to agree across Agent 3,
the orchestrator's startup sync, and any manual-ingest path. Two copies drift.
The citation-suggestion prompt was once duplicated between Agent 4 and the RAG
evaluator and the two silently diverged — the evaluation ended up scoring a
prompt the agent no longer used.

---

## 2. Orchestration

### 2.1 LangGraph, not plain Python or Google ADK

`orchestrate.py` runs agents 0–3 (plus an optional response step) as a
`StateGraph` with typed state and conditional edges.

**Why LangGraph.** The flow is a fixed sequence with two real branch points
("no seed paper found → stop", "neither extraction strategy found a reference
to fetch → stop") and a shared state object threaded through every node. That
is exactly a state machine.
LangGraph gives named nodes, conditional edges, streamed progress
(`stream_mode="updates"` drives the UI's step list), and a checkpointer, for
~150 lines.

**Why not Google ADK.** ADK is built for *LLM-driven* agents that decide their
own next action. Here the control flow is deterministic — an LLM deciding
"should I run Agent 2 now?" adds latency and a failure mode for no benefit.
ADK's dependency tree was also broken in the target environment (an `httpx`
version conflict via `google-genai`).

**Why not plain Python.** It was the first implementation. It works, but the
branch handling was ad-hoc `if` statements and there was no structured way to
stream progress to a UI or to resume from a checkpoint.

**Trade-off.** A dependency (`langgraph`) and a small amount of ceremony
(`TypedDict` state, node functions returning partial-state dicts).

### 2.2 One graph per run, `MemorySaver` checkpointer

The Streamlit app rebuilds the graph on each "Build corpus" click rather than
caching it.

**Why.** The checkpointer keys state by `thread_id` (the query string). Reusing
one graph across runs means a second run of the same query resumes from the
first run's checkpoint and can skip stages. A fresh graph per run starts clean;
cross-run resumption is instead handled by the agents' own on-disk state, which
is durable across process restarts (the in-memory checkpointer is not).

---

## 3. Discovery (Agent 0)

### 3.1 Provider order: arXiv → OpenAlex → Semantic Scholar

**Why arXiv first.** For a physics tool, the preprint is almost always what the
user wants, and the arXiv PDF link *never* 404s. OpenAlex's top-ranked hit is
frequently a publisher landing page that returns an HTML paywall with HTTP 200.

**Why OpenAlex second.** No rate limit, the best structured metadata, and it
surfaces open-access repository copies for published (non-preprint) work.

**Why Semantic Scholar last.** Its relevance ranking and abstracts are good, but
the keyless API pool returns HTTP 429 almost constantly. It is only useful with
a real key (`S2_API_KEY`).

### 3.2 Selection is download-verified, not field-verified

`find_and_fetch_seed()` walks every candidate of every provider and **downloads
until a PDF actually arrives** — a candidate with a `pdf_url` field whose
content is HTML, or 403s, is skipped and the walk continues to the next
candidate and then the next provider.

**Why.** An earlier version picked the first candidate that merely *had* a
`pdf_url` string, then failed the whole run when that URL turned out to be a
paywall. "Has a link" and "the link is a PDF" are different facts.

**Trade-off.** A failed query does several HTTP requests before giving up. The
`shared/fetch.py` downloader guards this: it checks the `%PDF-` magic bytes on
the first chunk and refuses non-PDFs immediately.

### 3.3 Manual-URL fallback

When nothing downloads, `discover_from_url()` / `--seed-url` takes an arXiv or
direct-PDF link. arXiv `/abs/` links are rewritten to `/pdf/`; the key becomes
`arxiv:<id>` or a URL hash.

**Why.** Search relevance is imperfect and some fields are genuinely paywalled.
The user often knows the exact paper. This keeps the pipeline usable instead of
dead-ending.

### 3.4 No-corpus fallback answer

If discovery still finds nothing and the caller wanted an answer (`--ask` / the
UI), a `fallback` node answers the query from the model's own knowledge,
prefaced with a note that it is ungrounded. The "supply a PDF link" prompt is
kept alongside it.

**Why.** A blank screen is a bad response to "I couldn't find a paper". The
model usually knows *something* about the topic, and saying so — clearly flagged
as not retrieved from sources — is more useful than stopping. The run still
exits non-zero and still asks for a link, so the degradation is visible.

### 3.5 `source_key` — deterministic document identity

Every reference and document folds to a prefixed key: `doi:…`, `arxiv:…`,
`pmid:…`, `url:<hash>`, `title:<hash>`, or `raw:<hash>`, in that order of
preference.

**Why.** The same paper is cited by several papers, downloaded once, ingested
once, and must be recognised as the same thing at each stage. The prefix tells a
consumer whether the identity is authoritative (a real DOI) or derived (a title
hash). Same input always produces the same key, so runs are comparable and
de-duplication is free. Title hashing strips stop-words and uses only the first
author's surname, because initials and subtitle wording are what differ between
two parses of one paper.

---

## 4. Ingestion (Agent 3)

### 4.1 Layout detection is optional; text-only is the fallback

`CITATION_LAYOUT_DETECTION` (default on locally, **off** on the deployment).
With it off, `process_pdf` extracts text blocks with PyMuPDF and skips figures
entirely.

**Why.** Detectron2 is not on PyPI, must be built from source against a specific
torch/CUDA build, and needs ~4 GB RAM for the model. That is a poor fit for a
container that should start fast and cheap. Text is 90% of what retrieval needs.
The layout stack's imports are deliberately kept inside the functions that use
them so the module (and its tests) import fine without them.

### 4.2 Figure VLM is opt-in; the caption is the default text

With layout on, every figure/table is still cropped and its **caption** kept as
the searchable text. A VLM description of each crop only happens if
`CITATION_FIGURE_VLM=1`.

**Why.** A VLM call per figure is the single most expensive thing in ingestion —
dozens of calls per paper, most of them describing plots that no query will ask
about. The caption already says what the figure *is*. The crop is saved with its
path in metadata, so a description can be generated on demand at query time for
the few figures that actually surface.

### 4.3 Semantic chunking, batched per page

Text on a page is concatenated and chunked in one `SemanticChunker` call rather
than one call per text block.

**Why.** `SemanticChunker` embeds every sentence to find breakpoints. Per-block
calls multiply that cost; per-page batching cuts embedding calls by ~10×.

### 4.4 Chunk id encodes position

ChromaDB ids are `chunk_0`, `chunk_1`, … assigned sequentially and never
deleted. The integer doubles as the row's position in the BM25 corpus and the
`texts` list.

**Why.** Hybrid search fuses a BM25 rank (array index) with a dense rank
(ChromaDB id). They must refer to the same chunk. Encoding position in the id
makes the mapping O(1) with no side table.

**Trade-off.** The invariant (contiguous from zero, no deletes) is load-bearing
but not enforced. `hybrid_search` drops any id outside the loaded range and logs
a warning rather than silently indexing the wrong chunk.

### 4.5 Per-document summary, written to a separate collection

One LLM call per paper produces a 120–180-word summary, stored in
`physics_summaries` (one row per document) — not in the main chunk collection.

**Why.** This is the stage-1 index for two-stage retrieval (§5). It must be
queryable independently of the chunks, and there is exactly one per paper, so a
separate collection is the natural shape. Cost is one call per paper, once, at
ingest — not per chunk, not per query.

---

## 5. Retrieval — two-stage

### 5.1 Summary shortlist → LLM gate → deep search

```
query ─▶ rank_documents()   nearest summaries (cheap vector search)
      ─▶ gate_documents()   LLM: "relevant prior work? Y/N" per summary
      ─▶ deep_search()      hybrid chunk search, restricted to survivors
```

**Why.** As the corpus grows, searching every chunk for every query gets slower
and noisier — a 200-paper corpus buries the 3 relevant papers' chunks among
thousands. Matching the *idea* against *paper-level summaries* first is a coarse
filter that costs one vector query plus a handful of tiny yes/no LLM calls, and
it bounds the expensive chunk search to a shortlist. This is the "control over
the database as it grows" the design calls for.

**The gate specifically.** Vector similarity on summaries is fuzzy — it will
rank a magnetohydrodynamics paper near a query about electronic transport
because both are "inverse problems in physics". A one-token LLM judgment
("could this be relevant prior work, even loosely?") removes those. It falls
back to the top-1 summary if it rejects everything, so a query never dead-ends.

**Trade-off.** `DOC_SELECT_K` (default 6) sequential LLM calls per query. On a
fast local model or a paid API this is sub-second; on a free hosted pool it can
add 30–60 s, so `CITATION_DOC_GATE=0` turns it off (keeping the similarity
shortlist).

### 5.2 `hybrid_search` gains a `doc_filter`

Stage 2 passes the shortlisted document names. The dense side uses a ChromaDB
`where={"document": {"$in": […]}}` filter; the sparse side post-filters the
BM25 top-k by document. The candidate pool widens (`max(60, k·10)` instead of
`max(15, k·3)`) because most global candidates get discarded by the filter.

**Why keep hybrid rather than dense-only for stage 2.** BM25 still matters for
exact-term matches (a specific method name, a material) that dense retrieval
blurs. Filtering both sides keeps that.

### 5.3 Hybrid search = BM25 + dense + Reciprocal Rank Fusion

RRF (`1/(k + rank)`, `k=60`) rather than score averaging.

**Why.** BM25 scores and cosine distances are not on comparable scales and their
distributions shift per query. RRF only uses rank position, so no normalisation
or tuning per query. `k=60` is the value from the original RRF paper and damps
the influence of a single retriever's #1.

### 5.4 Two outputs from one corpus

- **Related-work synthesis** (`research_answer`, the "Research a topic" flow):
  two-stage retrieval → `RESEARCH_CHAT_SYSTEM` synthesises "what has been done".
- **Citation insertion** (`agent4_assistant.suggest_citation`, the "Cite a
  draft" flow): flat hybrid search → rewrite the sentence with `\cite{key}`.

**Why two.** They have genuinely different goals. A literature overview wants
breadth across the shortlist; citing one sentence wants the single best-matching
passage. The citation flow stays flat because narrowing to a summary shortlist
first would over-constrain a single-sentence lookup.

---

## 6. Models & backends

### 6.1 `shared/llm.py` — one abstraction for chat, streaming and embeddings

Everything calls `chat()`, `chat_stream()` and `get_embeddings()`.
`LLM_BACKEND` picks `ollama` | `openai` for chat; `EMBED_BACKEND` picks
`ollama` | `openai` | `huggingface` for embeddings — the extra option exists
because a deployment can serve chat itself but call out to
`huggingface_hub.InferenceClient` for embeddings, or vice versa.

**Why.** The same code has to run two ways: fully local (Ollama daemon, nothing
leaves the machine) and hosted (no GPU, models behind an API). Without an
abstraction, every call site would branch. Provider SDKs are imported lazily
inside each backend so a local checkout doesn't need the `openai` package and a
hosted deploy doesn't need `langchain-ollama`.

**`chat_stream()` yields content deltas** (`str` chunks) with the same backend
switch as `chat()` — `ollama.chat(..., stream=True)` on one side,
`stream=True` and a `.choices[0].delta.content` read on the other, skipping the
`None` deltas the OpenAI API sends for role-only and terminal frames. Agent 7
(research chat) is built around it, and it is why no module outside
`shared/llm.py` needs to import `ollama` or `openai` directly — a bypass would
otherwise be tempting for a token-streaming UI.

**Embeddings are separately configurable** (`EMBED_BACKEND`) because a provider
that serves chat may not serve embeddings — e.g. the deployment uses the HF
router for chat but `huggingface_hub.InferenceClient` feature-extraction for
embeddings.

**The embeddings object implements the LangChain `Embeddings` interface**
(`embed_documents` / `embed_query`) so it drops straight into `SemanticChunker`.

### 6.2 Embedding model must stay fixed per corpus

Documented, not enforced: changing `CITATION_EMBED_MODEL` after a corpus exists
breaks search (dimension mismatch, or silently wrong geometry). The fix is to
delete `physics_vectordb` and re-ingest.

### 6.3 `config.py` is the only place constants live

Every model name, path, threshold, rate limit, and RRF constant is in
`config.py`, most overridable by environment variable. Paths anchor to
`CITATION_DATA_DIR` (default: `<project>/data`, separate from the source tree
— see §10.2).

**Why.** Changing a model or relocating the data store is a one-file edit, and
the deployment configures everything through env vars without touching code.

---

## 7. GROBID

### 7.1 External service, configurable, no longer a single point of failure

`GROBID_SERVER` defaults to `http://localhost:8070` — a local server is the
expectation, not a convenience fallback. Point it at a hosted instance (a
public Space, or the Cloud Run deployment's own GROBID service) by setting the
env var instead.

**Why not bundle it.** GROBID is a ~700 MB Java server with deep-learning
models, needs 4 GB RAM, and takes 30–60 s to start. Baking it into the app image
would bloat the image and slow every cold start, and running two servers in one
container is awkward on a scale-to-zero platform.

**Why a dead or unreachable server no longer stops the pipeline.** See §10.1 —
Agent 1 falls back to regex-based extraction per PDF whenever GROBID cannot be
reached or returns nothing usable, so an external Java service being down
degrades reference quality instead of ending the run.

### 7.2 `check_server=False` on the client

The GROBID client no longer pings `/api/isalive` on construction.

**Why.** A serverless GROBID (Cloud Run) may be cold-starting when the first
request arrives; the platform holds the request while it boots. A pre-check
would fail during that window. Real failures still surface per-file in the batch
result.

---

## 8. Deployment

### 8.1 One container image, env-configured per target

`Dockerfile` builds a single image (Streamlit app, honours `$PORT`). Local test
runs it with Ollama env vars; Cloud Run runs the same image with hosted-API env
vars.

**Why.** "Test what you deploy." A green local run of the image means the image
is good; only the backend wiring differs, and that is just environment.

### 8.2 Cloud Run, not Hugging Face Spaces

The initial target was an HF Space. It didn't work out:

- The free CPU Space tier was withdrawn.
- New Spaces default to ZeroGPU hardware, which only supports the Gradio SDK —
  incompatible with a Streamlit/Docker app, and the app needs no GPU anyway.

Cloud Run fits: it runs any container, honours `$PORT`, scales to zero (so an
idle demo costs nothing), and hosts GROBID as a second service the same way.

**Trade-off.** Requires a GCP project with billing enabled (free-tier limits
still apply). Cold starts (~seconds for the app, ~40 s for GROBID). The
container filesystem is in-memory, so the corpus is lost on scale-to-zero unless
a GCS volume is mounted at `CITATION_DATA_DIR`.

### 8.3 File logging off in the container

`CITATION_LOG_FILE=0` in the image.

**Why.** The app directory is read-only on Cloud Run, so creating `logs/` fails.
Container stdout is already captured by the platform's logging — a log file adds
nothing.

### 8.4 Streamlit, not Gradio or a custom frontend

**Why.** The app is form-in / structured-result-out with a long-running job that
needs progress streaming. `st.form` submits inputs atomically, `st.status`
renders the LangGraph step stream, and `st.session_state` keeps results across
tab switches. A custom frontend is weeks of work for a tool whose value is the
pipeline, not the UI.

---

## 9. Things deliberately not done

- **No database migrations.** Schema changes (adding the summaries collection)
  mean re-ingesting. Acceptable for a research tool with rebuildable corpora.
- **No auth / multi-tenant.** Single-user tool. The Cloud Run service is
  `--allow-unauthenticated` behind an obscure URL.
- **No async / job queue.** Ingestion runs synchronously in the request. Fine
  for one user; a shared deployment would need a queue.
- **`model_final.pth` (the layout checkpoint) is gitignored.** layoutparser
  downloads the PubLayNet weights on first use when the file is absent. In
  this checkout it is a symlink to the copy already sitting in `tech_ireland`,
  not a duplicated 830 MB file.
- **No Ragas evaluation harness yet.** `evaluate_rag.py` and
  `requirements-eval.txt` were deferred out of this merge (§10 below), not
  rejected — the prompts it would share with Agent 4 already live in
  `prompts.py`.

---

## 10. This merge's decisions

Two decisions made while combining `tech_ireland` (infrastructure) and
`citation_builder` (agents 5–7, tests) that neither source repo documents,
because neither source repo faced the question on its own.

### 10.1 Why Agent 1 keeps both extraction strategies

`tech_ireland` and `citation_builder` extract references two incompatible
ways, and each has the other's weakness. `tech_ireland` posts every PDF to
GROBID and parses the TEI: far richer output — per-reference title, authors,
year, DOI, and a confidence score for each consolidated DOI against the
printed reference — but it depends on a running external Java service, and
the old pipeline had a terminal `GROBID produced nothing → END` edge that
ended the whole run when that service was unreachable. `citation_builder` runs
`pdftotext` and matches numbered reference lines (`[1]`, `1.`, `(1)`): far
weaker output, a raw string per reference with no structured fields, but no
external dependency at all.

**Why keep both instead of picking the better one.** GROBID's structured TEI
is materially better input for Agent 2's DOI resolution — Agent 2 trusts
GROBID's DOI outright when the confidence score is high, and otherwise asks
Crossref using the parsed *title*, which is a far better query than a raw
reference string. Losing GROBID would mean losing that resolution quality
everywhere. But a single external Java service being the one thing standing
between "found a paper" and "produced nothing" is a fragile design for a
pipeline whose whole point is not dead-ending. So GROBID stays primary and the
regex path becomes a **floor, not a competitor**: tried only when GROBID is
down for the whole run, or — per PDF — when GROBID processed that PDF fine but
found no reference list in it. A paper GROBID could not parse is still covered
by the weaker path instead of being silently dropped, and a dead GROBID server
degrades the run's *reference quality* instead of stopping it outright.

**Provenance is recorded, not hidden.** Each `Reference` carries an
`extraction_method` (`"grobid"` or `"regex"`), and a per-run count sits under
`"extraction"` in `extracted_citations.json`. Nothing downstream is forced to
treat a raw-string reference the same as a fully structured one — a consumer
that cares can weigh them differently, and a reader of the JSON can see at a
glance how much of a given run leaned on the weaker path.

### 10.2 Why the ingestion manifest is a JSON file, not a collection scan

Before this merge, "has this PDF already been ingested?" had two sources of
truth, unioned on every call: an append-only, newline-delimited text file
living *inside* the ChromaDB directory (no deduplication, growing without
bound), and a full paginated scan of every metadata record in the collection,
5000 rows at a time.

**Why that was a problem.** Parsing a PDF — layout detection, chunking,
embedding, the per-paper LLM summary — is by a wide margin the most expensive
thing Agent 3 does. The already-ingested check exists specifically to avoid
redoing it. But the check itself ran on *every* ingest, for *every* candidate
PDF, and did so by reading every metadata record already in the corpus before
a single new PDF was processed. The check meant to make re-runs cheap was
making every run pay a cost that grows with the corpus, on top of whatever
actual parsing it went on to do.

**The fix.** `shared/manifest.py` replaces both sources of truth with
`data/ingested.json`, a single JSON object mapping `pdf_key → ingest
timestamp`, written atomically (temp file + `os.replace`, the same pattern
`shared/fetch.py` already uses for downloads so a crash mid-write can't corrupt
it). `load()` / `contains()` / `add()` / `add_many()` are all plain dict
operations — no collection round-trip on the common path. The collection scan
survives as `rebuild_from_collection()`, a repair function called on demand
when the manifest itself is lost or corrupted, rather than something every
ingest pays for.

This was only free to do *now*, before the corpus exists — a manifest
introduced against an already-populated collection would need a migration
step to seed itself from that collection's metadata first.

### 10.3 Seeding from an uploaded PDF, and why the query stopped being an identity

Agent 0's original job was "turn a research idea into a paper". Uploading
inverts that: you already have the paper, and the idea is optional. That is a
real entry point — you often start from something a colleague sent you, or from
a paper the automatic search cannot reach because it is not open access.

`discover_from_file()` handles it and deliberately joins the *existing* path
rather than paralleling it: it validates the magic bytes (`%PDF-`), reads a
title / DOI / arXiv id out of the file, derives the same kind of `source_key`
the rest of the system uses — `arxiv:` → `doi:` → a `file:<sha1>` content hash
when neither is present — writes the paper into `RAW_DIR` under the key-derived
name, and records a `SeedPaper`. Nothing downstream knows an upload happened;
Agent 1 finds a PDF in `RAW_DIR` exactly as it otherwise would.

**The subtlety this exposed.** `seed_papers.json` is keyed by query, which was
sound while the query was always something the caller typed — `discover()` and
`discover_from_url()` both rely on `seeds[query]` to answer "have I already
seeded this?". But on an upload with no query, the key is *inferred* from the
paper's title or filename stem, and two different papers can infer the same
string. The query then stops being an identity and becomes a label that happens
to collide, so a second upload silently overwrote the first's record — leaving
that paper in `RAW_DIR`, still extracted and ingested, but with no seed record
pointing at it.

Inferred queries are therefore disambiguated with the paper's own key; supplied
ones are left alone, because their stability is what the resume behaviour
depends on. The general lesson is one this codebase keeps relearning: a key
meaning "the request" and a key meaning "the thing" can look identical right up
until they diverge.

### 10.4 Ingestion reports how it extracted, for the same reason Agent 1 does

Layout detection is the most install-fragile part of the system, and
`process_pdf()` is deliberately forgiving of it — any failure degrades to
text-only extraction rather than aborting the run. That forgiveness has a cost:
the fallback is per-PDF and only logs a warning, so an entire corpus can come
out with no figures or tables and nothing in the return value says so.

That is the same shape of problem §10.1 solved for Agent 1, and it gets the
same answer. `process_pdf()` tags every entry with the mode that produced it
(`layout` or `text_only`) at the single point where the choice is made, and
`ingest_pdfs()` returns `{"extraction": {"layout": n, "text_only": n}}` counted
per document — mirroring the `"extraction"` block Agent 1 already writes into
`extracted_citations.json`. An unexpected fallback also logs one batch-level
summary, instead of leaving the evidence scattered across per-PDF warnings.

The tag rides on the corpus entries rather than the return type because
`process_pdf()` runs inside a `ProcessPoolExecutor`, where only the return value
crosses back from the worker. `upsert_corpus()` reads named keys, so the extra
one is inert.

A failed model load is remembered too. Because the failure is swallowed into a
text-only fallback, nothing upstream stopped the *next* document attempting the
same construction — a config that cannot load made every paper in a batch pay
for a full Detectron2 build that was guaranteed to fail. The failure is now
cached alongside successful models and re-raised without retrying.
