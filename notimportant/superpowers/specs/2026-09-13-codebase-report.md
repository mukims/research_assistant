# Marvin the Citebot / Citation Needed! — the codebase as it stands

*2026-09-13 · HEAD `2317a67` on `synthesis-depth` · 122 commits since 2026-09-07*

## 1. At a glance

| | |
|---|---|
| Package | `research_assistant/` — 11,142 lines of Python across `agents/` (9 agents + a GROBID controller), `shared/` (21 modules), `judgement/`, `config.py`, `prompts.py`, `schemas.py` |
| Entry points | `app.py` (Streamlit, 1,968 lines), `orchestrate.py` (LangGraph pipeline, 458), `watch.py` (directory daemon, 464) |
| Tests | 36 files, 10,398 lines; **597 passed, 1 skipped** at HEAD (`tests/test_ingestion.py`'s 44 deselected only because `data/ingest.lock` is held on this machine); CI runs Python 3.10 and 3.12 with the light `requirements-test.txt` plus an import sweep of every module |
| Corpus (this machine, index v2) | 308 documents, 16,336 chunks (13,436 text, 2,844 captions, 56 figure descriptions); median chunk 1,062 chars; 307 papers downloaded, 366 references unavailable (paywalled); 504 MB ChromaDB, 640 MB pulled PDFs |
| Models | chat/synthesis/judgement: `gemma4:e2b` on Ollama by default, **`gemini-3.5-flash-lite` when `GEMINI_API_KEY` is set** (as it is here); embeddings: `nomic-embed-text` on Ollama; GROBID 0.8.1 for PDF structure; an optional Detectron2/PubLayNet layout model (v1 only) |
| Docs | `README.md` (Marvin voice), `PIPELINE.md` (the pipeline reference, 310 lines), `ARCHITECTURE.md` (design rationale, 819 lines), `HOW_TO_USE.md` (rendered in Tab 5), `GEMINI.md` (operating invariants), `deploy/README.md`, two PDF guides |
| Planning | 11 specs and 12 plans under `notimportant/superpowers/`; four plans written and not yet built (§13) |

## 2. What it is

Two products sharing one corpus and one judgement engine:

1. **A literature pipeline.** Give it a research idea (or a paper): it finds a seed paper, mines the seed's reference list with GROBID, fetches the open-access references, indexes everything (chunks + one model summary per paper), and writes a synthesis of what the papers say about the idea.
2. **Citation verification.** For a sentence, a whole draft, or the seed paper's own in-text citations: retrieve what the cited paper says and judge whether it supports the claim, with an explicit rubric enforced in code.

Around them: a research chat ("brainstorming studio") over the corpus, a directory-driven daemon, a Streamlit front end with five tabs, and a GCP deployment. The README brands it *Marvin the Citebot*; the pitch calls it *Citation Needed!*; the deployed page still says *Research Assistant*.

## 3. Repository layout

```
app.py                 Streamlit UI — five tabs, sidebar telemetry, the container entry point
orchestrate.py         LangGraph graph: discover → ingest_seed → extract → fetch → ingest_refs → respond | fallback
watch.py               watchdog daemon: raw/ (run the pipeline), pulled_pdfs/ (ingest), drafts/ (cite) — cooldowns per kind
research_assistant/
  config.py            every path, model, backend and tunable; .env and DATA_DIR/.env are loaded; Gemini auto-selects the openai backend
  prompts.py           every prompt (21 constants) — citation, need-check, chat, lenses, figure description, summaries, gate, synthesis
  schemas.py           Reference · DownloadedPaper · SeedPaper — the inter-agent contracts, validated on load
  agents/
    agent0_discoverer  idea → seed PDF (arXiv, OpenAlex, Semantic Scholar; or a link; or an uploaded file)
    agent1_extractor   seed PDF → reference list via GROBID TEI (regex fallback), DOI confidence scoring
    agent2_fetcher     references → open-access PDFs (Unpaywall → Europe PMC → arXiv), retry policy, manifests
    agent3_ingestor    Agent 2's manifest → the shared ingestion path
    agent4_assistant   one sentence → a \cite{} suggestion with rationale
    agent5_batch_citer whole draft → cited draft + citation map + per-sentence report (batched need-check, aligned or aborted)
    agent6_manual_ingestor  a dropped PDF → the corpus
    agent7_research_chat    the brainstorming studio (condense, focus, memory, keys, lenses, suggestions)
    agent8_verifier    cited draft → every citation judged against its source; escalation; markdown + json reports
    grobid_controller  agent-level re-export of shared/grobid_manager
  shared/
    ingestion.py       process → upsert → mark → BM25 rebuild; the lock; summaries; v1 layout path; v2 dispatch
    extract.py  chunking.py  figures.py  ingest_v2.py  tokenize.py   — index v2: GROBID full text, sentence windows, crops, VLM
    search.py          hybrid BM25 + dense with RRF; doc_filter, exclude_types, expand_neighbours
    retrieve.py        two-stage retrieval: summary shortlist → batched gate → per-paper passages → map-reduce synthesis
    db.py              load_search_resources, cached per process on the BM25 pickle's mtime
    llm.py             chat / chat_stream / embeddings over ollama or any OpenAI-compatible endpoint; nomic prefixes on v2
    chat_context.py    the chat turn's mechanics: budget, keyed context, history fit, suggestions
    seed_audit.py      the uploaded paper's own in-text citations, judged; paywall deferral; upload-and-register
    pipeline_status.py telemetry file + callbacks + heartbeat + cooperative cancellation
    grobid_manager.py  GROBID lifecycle: Docker, JAR or remote; health; restart
    batch_uploader.py  multi-PDF / ZIP staging;   manifest.py  atomic.py  fetch.py  source_key.py  retry.py  log.py
  judgement/
    judge.py           the claim–evidence rubric: three slots, derived verdict, verbatim-span check, enforcement
    prompt.md          rubric V1.4 — rubric first, input last (a cacheable prefix)
    cases/cases.jsonl  6 regression cases (the prompt's own examples)
scripts/               index_stats.py · judge_bench.py · synth_compare.py
tests/                 36 files; unittest.TestCase collected by pytest; fakes at the model/network/index seams
deploy/                GCP: prereqs, VM, deploy, startup, allow-ip, idle auto-shutdown, vm-control
Dockerfile · Dockerfile.standalone · docker-compose.yml · pyproject.toml · requirements*.txt
data/                  all runtime state (gitignored; CITATION_DATA_DIR overrides; /mnt/disks/data on the VM)
```

## 4. Runtime and backends

- **Backend selection** (`config.py`): `LLM_BACKEND` is `ollama` unless `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set, in which case it becomes `openai` against Google's OpenAI-compatible endpoint with `gemini-3.5-flash-lite` as both `LLM_MODEL` and `CHAT_MODEL`; `OPENAI_BASE_URL` / `OPENAI_API_KEY` point any other OpenAI-compatible server. Embeddings stay on Ollama (`EMBED_BACKEND` defaults to `ollama` under Gemini) so the existing index is usable without re-embedding. `.env` in the project root and `DATA_DIR/.env` are loaded with `python-dotenv`.
- **Ollama options**: `CHAT_OLLAMA_OPTIONS = {"num_ctx": 4096}` (a GEMINI.md invariant for the CPU VM); `JUDGEMENT_OLLAMA_OPTIONS` and `SYNTHESIS_OLLAMA_OPTIONS` at 8,192 for the two calls that need it. Ignored by the openai path.
- **GROBID**: `GROBID_SERVER` (default `localhost:8070`); the manager can start it as a Docker container, a JAR, or treat it as remote; the sidebar exposes stop / restart / status / diagnostics. Reference extraction consolidates nothing (Crossref consolidation disabled in `2df6ec2` after 429 storms).
- **Index version**: `CITATION_INDEX_VERSION` (1 or 2) selects collection names, the BM25 pickle, the ingested manifest and the extraction path; v1 and v2 coexist on disk.
- **Temperatures**: judgement 0.0, synthesis 0.2, chat 0.3, condense/memory 0.0; the brief/board work in the queued plans uses 0.3–0.4.

## 5. The pipeline (Agents 0–3, the orchestrator, the daemon)

`orchestrate.py` is a LangGraph `StateGraph` with a `MemorySaver` checkpointer, rebuilt per run. Nodes and their on-disk state:

| node | agent | writes |
|---|---|---|
| `discover` | Agent 0: `discover(query)` walks arXiv → OpenAlex → Semantic Scholar until a PDF *actually downloads*; `discover_from_url`; `discover_from_file` (magic bytes, title/DOI/arXiv from the PDF, query inferred from the title when blank) | `data/raw/<key>.pdf`, `seed_papers.json` |
| `ingest_seed` | the shared ingestion path on the seed | the index |
| `extract` | Agent 1: GROBID batch → TEI → `Reference` records with DOI-confidence scoring; regex fallback per PDF; files PDFs under `raw/processed/` or `raw/failed/` | `extracted_citations.json` (+ `raw/grobid_output/*.tei.xml`) |
| `fetch` | Agent 2: de-duplicates references by `source_key`, resolves DOIs via Crossref when GROBID gave none, tries Unpaywall → Europe PMC → arXiv, checkpoints after every paper, retries transient failures on the next run | `pulled_pdfs/`, `downloaded.json`, `failed_downloads.json` |
| `ingest_refs` | Agent 3 → `ingest_pdfs` | the index, `ingested_v2.json` |
| `respond` / `fallback` | the synthesis (§6), then — if asked — the seed citation audit (§7); or an ungrounded answer when no seed was found | — |

Cooperative cancellation (`0a9a41e`): a `cancel_requested` flag in `pipeline_status.json`, set by the Stop buttons, checked at every node and at every paper / PDF / summary boundary; `PipelineCancelledError`. Telemetry: stage, step, item counts, the current item, a 15-event log, a 15-second heartbeat, stale-run detection by pid and age, thread-local and global progress callbacks — read by the sidebar every 3 s and by any browser session.

`watch.py` runs the same entry points from the file system: a PDF in `raw/` runs the pipeline for it, PDFs in `pulled_pdfs/` are ingested in batches, a `.txt` in `drafts/` is cited; a startup sync ingests anything missed.

## 6. Ingestion and retrieval

**Index v2** (`CITATION_INDEX_VERSION=2`, the local default now): `extract.py` sends the PDF to GROBID's `processFulltextDocument` (TEI cached under `raw/grobid_output/`), falls back to PyMuPDF with NFKC normalisation; `chunking.py` packs `pysbd` sentences into ~1,200-character windows (max 1,800) within sections with one sentence of overlap; every chunk carries `section`, `seq`, `page_first/last`, `extraction`; the embedded text is prefixed `Title: … Section: …` and, on v2 only, nomic's `search_document:` / `search_query:` prefixes. Figure and table captions become `caption` chunks always; crops are cut from GROBID coordinates; when the run's switch is on (`--describe-figures` / the Tab 1 toggle, one choice per run) a VLM description of every figure becomes a `figure_description` chunk marked as generated — never used as evidence by the verifier. One model summary per document goes to a separate summaries collection. BM25 v2 tokenises with NFKC + stopwords + Snowball; the pickle carries a header naming the tokenizer. **v1** (`LAYOUT_DETECTION`, Detectron2, semantic chunker) is untouched and still selectable.

**Search** (`search.py`): hybrid BM25 + dense (ChromaDB, cosine) fused by Reciprocal Rank Fusion (`k=60`); `doc_filter`, `exclude_types`, and `expand_neighbours` (adjacent chunks of the same document). Chunk ids `chunk_N` are contiguous and equal the BM25 row index — a load-bearing invariant. `db.load_search_resources` is memoised per process on the pickle's mtime.

**Two-stage retrieval and synthesis** (`retrieve.py`, built from the synthesis-depth plan): `rank_documents` over the summary collection → `gate_documents` (one batched YES/NO call over the shortlist, per-summary fallback on misalignment) → `dedupe_shortlist` by normalised title → keys `P1…Pn` → `passages_per_paper` (one restricted search per paper) → **map-reduce**: `paper_notes` per paper (establishes / method / limits, or *Not relevant*) then `synthesise` under three fixed headings (*What is established / Where the papers differ / The gap*), citing keys that `check_citation_keys` verifies; `single` mode is one call over the per-paper passages. `on_progress(stage, payload)` reports `shortlist`, `notes`, `synthesis`; the result carries `keys`, `notes`, `unverified_citations`, `irrelevant_cited`, `timings`.

## 7. Judgement and verification

- **The rubric** (`judgement/judge.py`, `prompt.md` V1.4): the model fills three slots — *finding*, *scope*, *strength* — each with an assertion and a verdict from a fixed vocabulary; `derive_judgement` aggregates them by a fixed rule into `Supports / Partially supports / Contradicts / Does not support / Unclear-insufficient`; `enforce_rubric` keeps the model's own verdict beside the derived one, records `rubric_mismatch`, checks the quoted `supporting_span` verbatim against the evidence, and caps confidence at *Medium* on any violation. Parse failures raise `JudgementParseError` after retries. The prompt puts the rubric first and the input last so Ollama can cache the prefix.
- **Agent 8** (`verify_draft`): splits the cited draft, pairs every `\cite{key}` with its claim (compound multi-cite sentences flagged), resolves the key to documents, retrieves within those documents (figure descriptions excluded), judges the top hit with its neighbours, and **escalates once** to the top-3 when the verdict says it saw too little; outcomes `judged / orphaned / unresolved / no_evidence / retrieval_failed / parse_failed / call_failed`; writes `_verification.json` and `.md`, with a *Not judged* total so the tiles add up.
- **The seed citation audit** (`shared/seed_audit.py`, 1,073 lines): from the seed's TEI, every sentence that carries a `<ref>` becomes a claim mapped to its bibliography entry; each claim whose cited paper is in the corpus is judged with the same rubric; the rest are *not downloaded*. `db9e7cd` added **paragraph-level paywall deferral** and a **direct upload workflow**: the audit lists the references it could not get (`get_deferred_missing_references`), Tab 1 offers a file uploader per missing reference, `save_and_register_reference_pdf` files the PDF under the reference's key and registers it in `downloaded.json`, and the audit can be re-run. Downloads: a markdown report (`generate_seed_audit_markdown`, with `explain_rubric_verdict` per item) and the JSON.
- **Judgement model**: `CITATION_JUDGEMENT_MODEL` if set, else the chat model — one tier.

## 8. Research chat — the studio (Agent 7, `6b9ca58`)

`ResearchChat.stream_turn` per turn: condense the follow-up into a standalone query (one call, temperature 0; shown as *Searched literature for*), global hybrid search at top-k×2 capped at 3 chunks per paper plus a focus search restricted to the papers cited last turn, `format_context` into `[S#]`-keyed blocks under 7,000 chars, a token budget against `num_ctx` (system + lens + memory + context + reserve; whole recent pairs kept, older ones folded into a rolling memory by a call), the answer streamed at 0.3, keys checked (unknown keys become warnings), sources marked cited, focus set, `[S#]` resolved to titles in history; `### 💡 Suggested Next Questions` parsed into chips. Five **lenses** (explore, gaps, contradictions, hypotheses, methodology) as a system-prompt addendum. Structured `turns` with sources, warnings, suggestions, mode; md/json export; a CLI REPL with `/mode /memory /sources /export`. Tab 4: lens selector, starters, evidence cards, a scratchpad of pinned snippets (session-state only), exports.

## 9. The UI (`app.py`)

Five tabs. **Research a topic**: upload (single → the full pipeline; several/ZIP → direct ingestion) or search; four toggles (answer, force, analyse figures, audit citations); live `st.status` progress driven by the status callbacks; the result — seed card, fetched / unavailable references, per-paper notes, the synthesis with its key legend and warnings, the citation audit with five tiles, filter tabs and downloads, the deferred-reference uploaders. **Cite a draft**: one sentence → suggestion, grounding, passages. **Cite a whole draft**: batch cite → cited text, download, key map, per-sentence report → *Verify citations* → five tiles, the report (downloadable). **Research chat**: the studio. **How to use**: `HOW_TO_USE.md` rendered. Sidebar: telemetry (stage, item, progress, elapsed, event log; Stop), corpus counts (cached), backend names, GROBID controls. Optional `APP_PASSWORD` gate. `tests/test_app_names.py` statically guards that every bare name the page calls is bound (lazy imports inside branches are the failure mode); `tests/test_app_ui_stress.py` (788 lines, new) exercises the audit-upload paths.

## 10. Data on disk

`data/` (or `CITATION_DATA_DIR`): `raw/` (seed PDFs, `grobid_output/`, `seed_audits/`, `processed/`, `failed/`), `pulled_pdfs/`, `drafts/`, `images/` (crops, 315 MB), `physics_vectordb/` (Chroma: `physics_papers`, `physics_summaries`, and their `_v2` twins), `bm25_index.pkl` / `bm25_index_v2.pkl`, `seed_papers.json`, `extracted_citations.json`, `downloaded.json`, `failed_downloads.json`, `ingested.json` / `ingested_v2.json`, `pipeline_status.json` (+ lock), `ingest.lock`, `logs/`, `eval/synthesis/` (the live comparison), `models/` (a MiniLM cross-encoder cached but unused — the reranker is deferred). Every manifest is written temp-then-replace (`shared/atomic.py`).

## 11. Tests and CI

597 passing tests plus 44 ingestion tests that run when the lock is free. Conventions: `unittest.TestCase` collected by pytest; heavy imports lazy so the tree imports with only `requirements-test.txt`; `CITATION_LOG_FILE=0`; models, network and the index are faked at their seams (`chat`, `chat_stream`, `hybrid_search`, providers, `process_pdf`, `judge`); live checks are manual scripts behind gates, never in the suite. Coverage is deep on ingestion bookkeeping, the fetcher's records, the extractor, the rubric and the verifier, pipeline status, the seed audit, the chat mechanics; thin on `app.py` (static guard + the stress file) and `watch.py`. **No accuracy measurement exists for the judge** — the six regression cases are the prompt's own examples.

## 12. Deployment

One GCE VM (`e2-custom-4-12288`, Debian 12, 50 GB persistent disk at `/mnt/disks/data`), Docker Compose with two containers — the app (port 8080, `Dockerfile` honours `$PORT`) and `grobid/grobid:0.8.1` (`-Xmx3g`) — a firewall rule for one caller IP, `allow-ip.sh`, an idle auto-shutdown script, `startup.sh` pulling secrets at boot. `Dockerfile.standalone` bundles `HOW_TO_USE.md` and sets `CITATION_INDEX_VERSION=2` (`227df5d`). The public URL (README) is `marvin-the-citebot.duckdns.org`, access-restricted. `deploy/README.md` still says the backend is OpenAI `gpt-4.1-mini` + `text-embedding-3-small` — the VM runs `gemma4:e2b` on Ollama per `HOW_TO_USE.md`, and Gemini is now the local default; the deploy doc is stale on this point.

## 13. Documents and the planning trail

`notimportant/superpowers/specs/` and `plans/` hold the design record: merged-research-assistant (2026-09-07), GCP deployment, judgement integration, rag-engine v2 (diagnosis, ingestion v2 with a build handoff and a verification record), judgement hardening, judge evaluation, synthesis depth, chat depth, narrated runs, brainstorm workspace, and the pitch gap analysis. Built from those plans: ingestion v2, judgement hardening (`e437ef4…9a26ae4`), synthesis depth (`2734de2…510d4ec`), and chat depth (subsumed by the studio). **Written, not built:** judge evaluation (the harness that would give an accuracy number), narrated runs (run transcripts + a live feed; carries a reconciliation note for the cancellation that landed meanwhile), brainstorm workspace (sessions, the board, moves, dives — a delta on the studio).

## 14. Known gaps, debts and inconsistencies

1. **Reliability judgement is absent** — no source assessment, no HIGH/MODERATE/LOW/UNRESOLVED policy, one model tier, no expert loop (see the pitch gap analysis). The verdict tables are relation tables.
2. **The judge is unmeasured** — the evaluation set is planned only.
3. **The paywall dominates the audit**: 366 of the corpus's references were unavailable; the audit's biggest tile is *not downloaded*. The upload path (`db9e7cd`) helps by hand; there is no abstract-level fallback.
4. **`app.py` is 1,968 lines** with the pipeline driver, three progress callbacks, the audit renderer and four tabs inline; the narrated-runs plan shrinks it but must be reconciled with the current Tab 4 and cancellation.
5. **Identity drift**: README says Marvin the Citebot, the page says Research Assistant, the pitch says Citation Needed!.
6. **`deploy/README.md` names a backend the VM does not run.**
7. **Chat budgets against `num_ctx` even on Gemini** (folds history at 4k; the brainstorm plan fixes this with `CHAT_WINDOW_TOKENS`).
8. **Two cancellation designs** on paper: `pipeline_status.cancel_requested` (built) and `runlog.cancel` (planned) — the plan's note says to build on the former.
9. Accepted losses recorded in the ingestion-v2 handoff: table cells, appendices, display formulas, vector-figure crops; 19 missing PDFs.
10. `main` is 57 commits behind `synthesis-depth` and 4 ahead of `origin/main`; nothing on the feature branches is pushed. A `feat/gcp-deployment` worktree exists under `.worktrees/`.
