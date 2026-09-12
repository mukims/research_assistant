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
paper. `ingest_pdfs()` marks every PDF it attempted — excluding any a worker
crashed on — once that batch's chunks are committed, so a crash partway through
re-parses the batch instead of recording work that never landed.

**Why.** Fetching 60 references takes minutes and hits flaky external servers;
ingestion runs layout detection and embedding per page. A process killed at 80%
must resume at 80%, not 0%. Processing is by far the most expensive step, so a
PDF that is processed but not marked would be re-processed on every subsequent
run forever.

**Every one of those writes is atomic.** They go through `shared/atomic.py` —
temp file in the same directory, `fsync`, then `os.replace` — so a reader sees
either the previous file or the new one, never a half-written one. That matters
more than it sounds: every reader of these files treats unparseable JSON as
"start from scratch", so a single torn write during a power cut would silently
discard a run's entire download history and fetch all of it again. The BM25
pickle goes through the same path, where a partial write is worse still —
`load_search_resources()` unpickles it on every query, so a torn one takes down
every search path at once.

**Resuming skips what is settled, not what merely failed.** A key in
`failed_downloads.json` is excluded from the next run only when its recorded
reason is a definitive one — paywalled, not indexed, 404. A transient reason (a
dropped connection, a rate limit, a 5xx, a truncated body) is retried, because
otherwise a ten-second network blip drops those references from the corpus
permanently, recoverable only by editing JSON by hand.

**Trade-off.** More disk writes, and a temp file alongside each one while it is
being written.

### 1.3 One implementation per capability, in `shared/`

PDF processing, chunking, ChromaDB upsert, hybrid search, source-key derivation,
durable file replacement, logging, retry — each lives once under `shared/` and
is imported by every agent that needs it.

**Why.** The check/mark bookkeeping in ingestion has to agree across Agent 3,
the orchestrator's startup sync, and any manual-ingest path. Two copies drift.
The citation-suggestion prompt was once duplicated between Agent 4 and the RAG
evaluator and the two silently diverged — the evaluation ended up scoring a
prompt the agent no longer used.

### 1.4 One ingest at a time

`ingest_pdfs()` holds a lock for the whole of its work: a re-entrant
`threading.RLock` against other threads in the process, and an `flock` on
`data/ingest.lock` against other processes sharing a `DATA_DIR`.

**Why.** Two ingests running at once corrupt the corpus rather than merely
slowing each other down. `upsert_corpus()` allocates chunk ids by reading the
collection's current maximum and counting up from it, so overlapping writers
mint the same ids and one set of chunks is dropped. `rebuild_bm25()` rewrites a
single pickle, so overlapping writers can leave an index that will not load at
all. The ingestion manifest is a read-modify-write, so overlapping readers lose
each other's updates.

This is not a hypothetical race. `watch.py` drives the `raw/` pipeline on one
thread while its `pulled_pdfs/` watcher drains on another — and Agent 2 writes
its downloads *into* `pulled_pdfs/`, so fetching a seed's references arms that
second thread on every run. `app.py` and `watch.py` can also be pointed at one
`DATA_DIR` simultaneously.

**Trade-off.** A second ingest waits rather than failing, and parsing a batch
can take minutes. It logs that it is waiting on the lock rather than blocking
silently.

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
but only partly enforced. Serialising ingestion (§1.4) closes the way two
concurrent writers could mint the same id; nothing stops a manually deleted row
from opening a gap. `hybrid_search` drops any id outside the loaded range and
logs a warning rather than silently indexing the wrong chunk.

### 4.5 Per-document summary, written to a separate collection

One LLM call per paper produces a 120–180-word summary, stored in
`physics_summaries` (one row per document) — not in the main chunk collection.

**Why.** This is the stage-1 index for two-stage retrieval (§5). It must be
queryable independently of the chunks, and there is exactly one per paper, so a
separate collection is the natural shape. Cost is one call per paper, once, at
ingest — not per chunk, not per query.

### 4.6 A chunk's identity is `(document, text)`, not `text`

Re-ingesting a paper must not double its chunks, so `upsert_corpus()` skips text
it has already stored. That check is keyed on the document *and* the text, and
compares the text in the truncated form that actually reaches ChromaDB.

**Why both halves.** Keyed on text alone, the second of two papers sharing a
sentence loses it — and boilerplate makes that common: shared method
descriptions, standard equations, "the remainder of this paper is organized as
follows". The passage then survives only under the first paper's
`citation_source`, so retrieving it cites the wrong work. For a tool whose
output is citations, silently attributing a passage to the wrong paper is the
worst failure it can have.

**Why the truncated form.** `collection.add()` stores `text[:EMBED_MAX_CHARS]`.
Comparing the full in-memory text against that stored prefix never matches for a
long chunk, so every chunk over the limit was re-inserted on each run and the
corpus grew a fresh copy of it every time.

### 4.7 v2: GROBID full text, then sentence windows

Under `CITATION_INDEX_VERSION=2` every PDF goes through GROBID's
`processFulltextDocument` (TEI cached in `raw/grobid_output/`), and the
chunker packs `pysbd` sentences into ~1,200-character windows within a
section, with one sentence of overlap. PyMuPDF is the fallback, recorded per
chunk as `extraction=pymupdf`.

**Why.** On the v1 corpus the median chunk was 242 characters and 45% were
under 200 — chopped bibliography entries and font-map garbage, which BM25's
length normalisation ranks above real passages. GROBID keeps the
bibliography out (the heading heuristic found it on 1 of 3 papers; GROBID
on 4 of 4), de-hyphenates (`quan- tum`), normalises ligatures (`ﬁeld`, in
31% of v1 chunks and invisible to a query for "field"), and yields sections
and captions PyMuPDF cannot. Windows give the 4k-window generator ~1,800
tokens of evidence from six chunks instead of ~400.

**Trade-off.** 6–11 s of GROBID per paper at ingest; formulas are dropped
from paragraph text (they were glyph soup inline); a broken-font PDF is
garbage under both paths and is dropped by the alphabetic-ratio filter.

### 4.8 What is embedded is not what is stored

The stored chunk is the window verbatim — quoted, cited, verified. The
embedded text is `search_document: Title: … Section: … <window>`: nomic's
task prefix plus a contextual header. Queries get `search_query:`. Prefixes
apply to the v2 index only; v1's vectors were made without them and mixing
is worse than neither.

### 4.9 Figures: GROBID captions, coordinate crops, descriptions as a run's choice

Captions come from GROBID's `<figDesc>` for figures *and* tables (v1's
`find_caption` had no table branch — all 548 tables were placeholders — and
attached body text to 30% of figures). Crops are PyMuPDF renders of GROBID's
`<graphic coords>` box; no Detectron2. Whether every figure is then described
by the model is one switch per run (`--describe-figures`, the Tab 1 toggle,
default `CITATION_FIGURE_VLM`) — never a per-image choice — and the
description is a separate `figure_description` chunk whose text opens with
"Auto-generated description of … verify values against the figure".

**Why the marker and the exclusion.** In testing, the model placed a plot's
peaks at ±0.5 where the axis read ±1.0. Descriptions are retrievable for
synthesis and chat; Agent 8 excludes them as evidence (`exclude_types`), so
a citation verdict rests on the paper's text.

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
— see §11.2).

**Why.** Changing a model or relocating the data store is a one-file edit, and
the deployment configures everything through env vars without touching code.

---

## 7. GROBID

### 7.1 External service, configurable, no longer a single point of failure

`GROBID_SERVER` defaults to `http://localhost:8070` — a local server is the
expectation, not a convenience fallback. Point it at a hosted instance (a
public Space, or the GCE deployment's own GROBID service) by setting the
env var instead. Running one locally via Docker is the standard route, and is
walked through step by step in
[Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).

**Why not bundle it.** GROBID is a ~700 MB Java server with deep-learning
models, needs 4 GB RAM, and takes 30–60 s to start. Baking it into the app image
would bloat the image and slow every cold start, and running two servers in one
container is awkward on a single-container host.

**Why a dead or unreachable server no longer stops the pipeline.** See §11.1 —
Agent 1 falls back to regex-based extraction per PDF whenever GROBID cannot be
reached or returns nothing usable, so an external Java service being down
degrades reference quality instead of ending the run.

### 7.2 `check_server=False` on the client

The GROBID client no longer pings `/api/isalive` on construction.

**Why.** A containerized GROBID may be initializing when the first
request arrives. A pre-check would fail during that window. Real failures still surface
per-file in the batch result.

---

## 8. Deployment

### 8.1 One container image, env-configured per target

`Dockerfile` builds a single image (Streamlit app, honours `$PORT`). Local test
runs it with Ollama env vars; the GCE deployment runs the same image with hosted-API env
vars.

**Why.** "Test what you deploy." A green local run of the image means the image
is good; only the backend wiring differs, and that is just environment.

### 8.2 A GCE VM, not Cloud Run

The initial target was a Hugging Face Space, then Cloud Run. Neither survived
contact with the corpus.

- The free CPU Space tier was withdrawn, and new Spaces default to ZeroGPU
  hardware, which only supports the Gradio SDK.
- **Cloud Run cannot hold this corpus.** It has no persistent disk; the only
  durable option is a GCS FUSE mount, and the corpus is `chroma.sqlite3` plus
  mmap'd HNSW `.bin` files. GCS FUSE has no POSIX file locking and turns small
  random writes into whole-object rewrites. SQLite and a memory-mapped vector
  index on that substrate is corruption, not slowness.

A GCE VM with a persistent disk is an ordinary filesystem, so ChromaDB, the
BM25 pickle and the ingestion lock all work untouched — **no application code
changes are needed for persistence**. `e2-custom-4-12288` (4 vCPU, 12 GB):
GROBID takes two cores during extraction at `GROBID_BATCH_CONCURRENCY = 2`, and
four cores keep the UI responsive while it does.

**Trade-off.** Always-on cost (≈$95/mo) instead of scale-to-zero, and a machine
to patch. Accepted: a scale-to-zero design that corrupts its corpus is not
cheaper, it is broken.

**Rejected.** Filestore gives real NFS locking but starts at 1 TiB ≈ $200/mo.
Cloud SQL + pgvector removes Chroma but rewrites `db.py`, `retrieve.py` and
`ingestion.py`, and leaves the BM25 pickle homeless.

### 8.3 Access: one firewall rule, not IAP

`tcp:8080` from a single `/32`. IAP and Google-managed certificates both require
a domain, which is not available. An unauthenticated app nobody can route to is
not an exposed app — and the app spends an OpenAI key, so this is a billing
control as much as a privacy one.

**Why this needs a script.** Pinning to one IP means presenting from an
unfamiliar network silently locks you out: the app is up, healthy, and
unreachable. `deploy/allow-ip.sh` takes no arguments and repoints the rule at
your current address.

**Trade-off.** Plain HTTP: the firewall protects the key's effects, but the
session is unencrypted in transit. Acceptable for one user on one address. With
a domain, the upgrade is IAP + a managed cert on an HTTPS load balancer
(~$18/mo) and nothing else changes.

### 8.4 File logging off in the container

`CITATION_LOG_FILE=0` in the image.

**Why.** Container stdout is captured by Docker and the platform's logging — an internal log file adds nothing and clutters the volume.

### 8.5 Streamlit, not Gradio or a custom frontend

**Why.** The app is form-in / structured-result-out with a long-running job that
needs progress streaming. `st.form` submits inputs atomically, `st.status`
renders the LangGraph step stream, and `st.session_state` keeps results across
tab switches. A custom frontend is weeks of work for a tool whose value is the
pipeline, not the UI.

---

## 9. Verification (Agent 8)

### 9.1 An audit pass, not an inline gate

Agent 8 runs after Agent 5, over its output. Agents 4 and 5 are unchanged.

**Why.** Judgement is one claim–evidence pair at a time; the slot
decomposition is what makes it accurate and it cannot be batched without
losing that. Agent 5 already batches its citation-need check
(`CITATION_CHECK_BATCH_SIZE = 20`) because per-sentence calls were too
expensive, and putting judgement inline on every retrieved candidate roughly
triples a batch run's call count — acceptable against a hosted API,
impractical against a local CPU model.

**The trade-off.** A wrong citation is inserted and then flagged rather than
blocked, and the draft is not corrected automatically. Making judgement a gate
is a coherent future change; it is not this one.

### 9.2 Evidence is re-retrieved, not replayed

Agent 5 does not persist the chunk text it cited — `citation_entries` records
only source metadata, and `_report.md` is prose. So the evidence behind a
citation is not recoverable from Agent 5's output, and Agent 8 re-runs
`hybrid_search` restricted to the cited source.

**Why that is acceptable.** The question becomes "does this source's strongest
evidence for this claim support it?" If even the best chunk fails, the citation
is wrong regardless of what Agent 5 saw. The failure runs in the safe
direction: re-retrieval cannot manufacture support the source does not
contain. The report quotes the chunk that was judged under **Evidence
judged** on every flagged citation (truncated to 400 characters), and the JSON
record carries it in full for every citation, so the reader can always see
which text the verdict rests on — including for `Does not support`, where
`supporting_span` is legitimately `null` and the quoted chunk is the only
evidence text there is.

The record's `sentence_index`, `claim` and `evidence` are the pipeline's own,
never the model's: only the rubric's six fields are merged out of a reply, so
an echoed key in the model's JSON cannot rewrite what was actually judged.

### 9.3 The prompt is code

`judgement/prompt.md` is a 259-line rubric whose aggregation order and
scope-versus-contradiction distinction are load-bearing, and no unit test
covers them. `judgement/cases/cases.jsonl` is the only guard. Run the live
suite (`RUN_LLM_TESTS=1`) before shipping any prompt edit.

### 9.4 The sentence boundary survives citation

`gemma4:e2b` drops the terminal full stop when it appends `\cite{}`. Agent 5
restores it (`_restore_terminal_punctuation`), because the draft is joined
with spaces and Agent 8 splits on `[.!?]` + whitespace: a lost stop merged
two sentences into one claim, and both citations were judged against it.
Agent 8 still flags any sentence with ≥2 keys and >40 words as
`compound_sentence` — the signature of a merge that got through.

### 9.5 The rubric is enforced in code

The model's reply is evidence, not verdict. `enforce_rubric()` derives the
aggregate from the three slot verdicts by the rubric's own Step-3 rules,
records the model's stated judgement as `model_judgement` and any
disagreement as `rubric_mismatch`, lists slot verdicts outside their
vocabulary in `rubric_violations` (treated as *Insufficient* for
aggregation), and checks `supporting_span` verbatim against the evidence
(`span_verified`). Any of those caps `confidence` at Medium.

**Why derive rather than reject.** A rejected reply is retried at
temperature 0 — the same reply — and the citation ends up unjudged. A derived
verdict with the disagreement on record is more useful, and it is what the
evaluation set (next) will measure.

### 9.6 Evidence: the top hit with its neighbours, then a second look

Agent 8 retrieves `max(JUDGEMENT_TOP_K, JUDGEMENT_ESCALATE_TOP_K)` hits from
the cited paper once, wraps each in its adjacent chunks (`expand_neighbours`),
and judges the top `JUDGEMENT_TOP_K`. If that verdict reports it saw too
little — `evidence_sufficiency` not *sufficient*, or *Unclear* / *Does not
support* — it judges once more on the top `JUDGEMENT_ESCALATE_TOP_K`, capped
at `JUDGEMENT_EVIDENCE_MAX_CHARS`. The second verdict stands; the first is
kept (`first_judgement`, `first_sufficiency`, `escalated`). At most two calls
per citation. `figure_description` chunks are never evidence.

**Why.** `hits[0]` alone reported *strength: Insufficient* on a claim whose
supporting paragraph sat one chunk away in the same paper. The rubric's
sufficiency field exists to separate a retrieval gap from a citation
failure; now it drives one.

### 9.7 The rubric as a cacheable prefix

`judgement/prompt.md` puts the rubric and worked examples first and the `## Input` block (`{{CLAIM}}`, `{{CITATION_EVIDENCE}}`) last. On Ollama (`gemma4:e2b`), moving the variable input to the end gives Ollama a 3.7k-token invariant prompt prefix to cache across judgements. Benchmarking the regression suite (`scripts/judge_bench.py`) confirmed accuracy is preserved (6/6 pass) while warm median latency dropped by 35% (68.8s baseline → 44.4s reordered, max 128.5s → 45.9s).

---

## 10. Things deliberately not done

- **No database migrations.** Schema changes (adding the summaries collection)
  mean re-ingesting. Acceptable for a research tool with rebuildable corpora.
- **No auth / multi-tenant.** Single-user tool. The VM is firewalled to
  the operator's single /32 IP address.
- **No async / job queue.** Ingestion runs synchronously in the request. Fine
  for one user; a shared deployment would need a queue.
- **`model_final.pth` (the layout checkpoint) is gitignored.** layoutparser
  downloads the PubLayNet weights on first use when the file is absent. In
  this checkout it is a symlink to the copy already sitting in `tech_ireland`,
  not a duplicated 830 MB file.
- **No Ragas evaluation harness yet.** `evaluate_rag.py` and
  `requirements-eval.txt` were deferred out of this merge (§11 below), not
  rejected — the prompts it would share with Agent 4 already live in
  `prompts.py`.

---

## 11. This merge's decisions

Two decisions made while combining `tech_ireland` (infrastructure) and
`citation_builder` (agents 5–7, tests) that neither source repo documents,
because neither source repo faced the question on its own.

### 11.1 Why Agent 1 keeps both extraction strategies

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

### 11.2 Why the ingestion manifest is a JSON file, not a collection scan

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

### 11.3 Seeding from an uploaded PDF, and why the query stopped being an identity

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

### 11.4 Ingestion reports how it extracted, for the same reason Agent 1 does

Layout detection is the most install-fragile part of the system, and
`process_pdf()` is deliberately forgiving of it — any failure degrades to
text-only extraction rather than aborting the run. That forgiveness has a cost:
the fallback is per-PDF and only logs a warning, so an entire corpus can come
out with no figures or tables and nothing in the return value says so.

That is the same shape of problem §11.1 solved for Agent 1, and it gets the
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
