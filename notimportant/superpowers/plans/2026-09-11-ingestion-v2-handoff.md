# Ingestion v2 — build handoff

*2026-09-11. For the agent building this. Read this file first, then the two documents it points at.*

## What you are building

A second index for Marvin the Citebot, selected by `CITATION_INDEX_VERSION=2`, that fixes how papers are turned into chunks:

- **GROBID full-text extraction** instead of raw PyMuPDF page blocks — sections, clean text (no `quan- tum`, no `ﬁeld`), bibliography kept out, figure/table captions with the bitmap's coordinates.
- **Sentence-window chunks** (~1,200 chars, one-sentence overlap, never across a section) instead of per-page semantic chunks whose median was 242 chars and 45% were under 200.
- **Figures and tables as first-class chunks**: a caption chunk always; a crop from GROBID's box; and — when the user switches it on for a run — a `gemma4:e2b` description of every figure, stored as its own chunk marked as generated.
- **A BM25 tokenizer that sees the words the reader sees** (NFKC + stemming), and **nomic task prefixes** on the embedder.

The v1 index keeps working untouched. An env var picks which index the app uses.

## The two documents that govern the work

| Document | Role |
|---|---|
| `notimportant/superpowers/specs/2026-09-11-rag-engine-v2-design.md` | **The spec.** Binding authority. §3.5, §3.6 and §4 (all of it, §4.6 in particular) are in scope; §2, §3.1–3.4 and §5 are **not** — do not build them. |
| `notimportant/superpowers/plans/2026-09-11-ingestion-v2.md` | **The detailed plan.** 16 tasks, each with the failing test, the code, the run command and the commit message. Follow it task by task. Where this handoff and the plan disagree, the plan wins; where the plan and the spec disagree, the spec wins. |

Two pieces of the plan were run against their own tests before the plan was written and pass: the chunker (Tasks 6–7, 20/20 with and without `pysbd`) and the TEI parser (Task 4). Transcribe them; don't redesign them.

## Environment facts

- **Interpreter:** `/home/shardul/miniconda3/envs/ml/bin/python` — has chromadb, fitz, lxml, ollama. Install `pysbd==0.3.4` and `snowballstemmer==3.1.1` into it (Task 2). Do not `pip install -e .` anywhere else.
- **Tests:** `CITATION_LOG_FILE=0 python -m pytest tests/ -v` (that interpreter). `unittest.TestCase` classes; pytest collects them.
- **GROBID:** `http://localhost:8070` (`curl localhost:8070/api/isalive` → `true`). Needed only for Task 16.
- **Ollama:** `localhost:11434`, models `gemma4:e2b` and `nomic-embed-text`. Needed only for Task 16.
- **Corpus:** `data/` in this checkout — 323 docs in the v1 index, 304 PDFs on disk (`data/pulled_pdfs/`), 16 more with a cached TEI only (`data/raw/grobid_output/`).
- **Branch:** create `ingestion-v2` from `main` before Task 1. Commit after every task with the plan's message. **Never push. Never merge.** The human decides both.

## Rules that must hold at every commit

1. **v1 is read-only.** With `CITATION_INDEX_VERSION=2`, nothing may open `physics_papers`, `physics_summaries`, `data/bm25_index.pkl` or `data/ingested.json` for writing. `data/` is never wiped, never `rm -rf`'d, never "cleaned up".
2. **Defaults change nothing.** `CITATION_INDEX_VERSION` defaults to `1`, `CITATION_FIGURE_VLM` to `0`. The app started with no env vars behaves as today. (One deliberate exception: Task 3 — the next v1 ingest rebuilds its BM25 pickle with the new tokenizer. That is the ligature fix and is wanted.)
3. **Nothing deploys.** `deploy/`, `Dockerfile`, `Dockerfile.standalone`, `docker-compose.yml`, `HOW_TO_USE.md` are not touched. The VM is not touched.
4. **Heavy imports stay inside functions.** `chromadb`, `fitz`, `requests`-to-GROBID, `pysbd` are imported lazily. CI's import sweep runs with only `requirements-test.txt` installed and must pass.
5. **No provider SDK outside `shared/llm.py`.** No `import openai`, no `import ollama` anywhere else. Descriptions go through `shared.llm.chat(..., images=[...], temperature=0.0)`.
6. **Chunk ids stay `chunk_N`, contiguous, never deleted.** `hybrid_search` aligns BM25 rows to them.
7. **Figure analysis is one switch per run** — CLI flag, pipeline state, Tab 1 toggle. Never a per-image choice, never a background/deferred pass.
8. **Tests before code, per task**, exactly as the plan's steps say. Run the named test file before committing; run the full suite at Tasks 14 and 16.

## The order

Foundations → pure modules (no services needed) → glue → user-facing switch → docs → one live run. Each task ends with green tests and a commit.

### Stage A — foundations

| # | Task | Files | Done when |
|---|---|---|---|
| 1 | Index version + v2 settings | `config.py`; new `tests/test_config_version.py` | `CITATION_INDEX_VERSION=2` yields `physics_papers_v2`, `physics_summaries_v2`, `bm25_index_v2.pkl`, `ingested_v2.json`; default yields today's names. `CHUNK_TARGET_CHARS=1200`, `CHUNK_MAX_CHARS=1800`, `CHUNK_MIN_CHARS=200`, `CHUNK_MIN_ALPHA=0.6`, `GROBID_TEI_DIR`, `GROBID_FULLTEXT_TIMEOUT=300` exist. |
| 2 | BM25 tokenizer | new `shared/tokenize.py`; `requirements.txt`, `requirements-test.txt`; new `tests/test_tokenize.py` | `tokenize("The Fluctuations of the fields") == ["fluctuat", "field"]`; `tokenize("ﬁnite-size eﬀects") == tokenize("finite-size effects")`; `TOKENIZER_VERSION == "v2"`; `legacy_tokenize` reproduces the old `\w+`. |
| 3 | Versioned BM25 pickle | `shared/ingestion.py` (`rebuild_bm25`), `shared/db.py`, `shared/search.py`; new `tests/test_db.py`, extend `tests/test_search.py` | Pickle is `{"tokenizer","built_at","bm25"}`; loader accepts both formats and tags `bm25.tokenizer_version`; `hybrid_search` tokenises the query to match; title (`citation_source`) is indexed with each chunk. |

### Stage B — pure modules (fixtures only; no GROBID, Ollama or Chroma)

| # | Task | Files | Done when |
|---|---|---|---|
| 4 | TEI → `Document` | new `shared/extract.py`; new `tests/fixtures/sample.grobid.tei.xml`; new `tests/test_extract.py` | `parse_tei()` returns `Document(key, title, abstract, sections[Section(heading, kind, paragraphs[Paragraph(text, page)])], figures[Figure(id, kind, label, caption, page, bbox, refs)], extraction="grobid", n_bib)`. Pages 0-based from `coords`. Running headers (`(3 of 11)`) fold into the previous section. `<formula>` is not a paragraph. Figure bbox from `<graphic coords>`, else the union of the figure's coords. `<ref type="figure">` sentences attach to their figure. Use `"".join(el.itertext())` — not `" ".join` — when reading element text. |
| 5 | `extract()` with cache + fallback | `shared/extract.py`; extend `tests/test_extract.py` | Cached TEI (any of Agent 1's suffixes) is reused without calling GROBID; a fresh response is cached to `GROBID_TEI_DIR/<stem>.grobid.tei.xml`; GROBID failure, timeout, malformed or empty body → `extract_pymupdf()` (NFKC, repeated blocks on ≥30% of pages dropped, cut at the last `References` heading, one section of kind `other`, no figures, `extraction="pymupdf"`). |
| 6 | Sentences + windows | new `shared/chunking.py`; new `tests/test_chunking.py` | `sentences()` keeps `Phys. Rev. B …` and `Ref. [3]` whole; `pack_windows()` closes at target, never exceeds max, carries one sentence of overlap, prefers a paragraph boundary at ≥70% of target, gives a >max sentence its own window, never emits the overlap alone. |
| 7 | Filters + `chunk_document()` | `shared/chunking.py`; extend `tests/test_chunking.py` | Acknowledgements/funding dropped; paragraphs with alphabetic ratio <0.6 dropped; sections <200 chars merged forward (trailing one backward); `Chunk(text, type, section, section_raw, page_first, page_last, seq, figure_id, figure_kind, figure_label)`; text windows <200 chars dropped, captions exempt; `seq` contiguous over text then captions. |
| 8 | Figures + prompt | new `shared/figures.py`; `prompts.py`; new `tests/test_figures.py` | `crop_figure()` renders the bbox +10 pt pad, clamped to the page, via PyMuPDF `clip`; `figure_context()` = `Caption: …` + `Referenced in the text:` lines (≤4); `describe_crop()` calls `shared.llm.chat` with `images=[path]`, `temperature=0.0`; `DESCRIPTION_PREFIX = "Auto-generated description of {label} — verify values against the figure: "`; `FIGURE_DESCRIPTION` rewritten to describe qualitatively and quote numbers only when legible. |

### Stage C — glue (no behaviour change without the switch)

| # | Task | Files | Done when |
|---|---|---|---|
| 9 | `process_pdf_v2()` | new `shared/ingest_v2.py`; new `tests/test_ingest_v2.py` | Emits dicts with `document, citation, page, type, content, extraction` plus `embed_text` (`"Title: {title}. Section: {heading}. " + text`) and flat `meta` (`section, section_raw, page_first, page_last, seq, extraction`, and for captions `figure_id, figure_kind, figure_label, image_path, described`). Caption chunk always; crop always attempted; `figure_description` chunk only when `describe_figures=True` and a crop exists, with `meta.extraction="vlm"` and content prefixed by `DESCRIPTION_PREFIX`; a failed description leaves `described=False` and continues; one `summary_source` entry per document (abstract + introduction + conclusion, capped at `SUMMARY_MAX_CHARS`). |
| 10 | `upsert_corpus()` accepts pre-chunked entries | `shared/ingestion.py`; extend `tests/test_upsert.py` | Types `text_chunk`/`caption`/`figure_description` bypass the semantic chunker; `embed_text` is embedded, `content` is stored; `meta` merged unprefixed; `summary_source` sets that document's summary input and is not stored; v1 entries still go through the chunker; `upsert_summaries` embeds with `embed_documents`. All existing upsert tests still pass. |
| 11 | Version dispatch + the run switch | `shared/ingestion.py` (`process_pdf`, `ingest_pdfs`, `_ingest_pdfs_locked`); extend `tests/test_ingestion.py` | `process_pdf(..., describe_figures=None)` → v2 path when `INDEX_VERSION >= 2`, unchanged otherwise; `ingest_pdfs(..., describe_figures=None)` with `None → config.FIGURE_VLM`, resolved once and passed to every `process_pdf` call (including the process-pool path); `result["extraction"]` counts `grobid`/`pymupdf` under v2; `result["described"]` counts description entries. |
| 12 | nomic prefixes | `shared/llm.py`; extend `tests/test_llm.py` | `get_embeddings()` wraps the backend in `_PrefixedEmbeddings` (`search_document: ` / `search_query: `) only when `INDEX_VERSION >= 2` **and** the model name starts with `nomic-embed`. |
| 13 | Agent 8 excludes VLM text | `shared/search.py`, `agents/agent8_verifier.py`; extend `tests/test_search.py`, `tests/test_verifier.py` | `hybrid_search(..., exclude_types=None)` drops matching `metadata["type"]` on the sparse side and adds `{"type": {"$nin": [...]}}` on the dense side (`$and` with the document filter; `None` when nothing is filtered); the verifier passes `exclude_types={"figure_description"}`. |

### Stage D — user-facing

| # | Task | Files | Done when |
|---|---|---|---|
| 14 | The switch everywhere | `agents/agent3_ingestor.py`, `agents/agent6_manual_ingestor.py`, `orchestrate.py`, `app.py`; extend `tests/test_orchestrate_upload.py` | `--describe-figures` (store_true, default `None`) on Agent 3, Agent 6 `--once`, and `orchestrate.py`; `PipelineState.describe_figures`; `ingest_seed` and `ingest_refs` pass `state.get("describe_figures")`; Tab 1 gets a third toggle *"Analyse figures and tables with the model"* (default `config.FIGURE_VLM`) in **both** forms, passed to `ingest_pdfs` in the batch path and into the pipeline inputs; `st.write` of the described count. Full suite green; `python -c "import app, orchestrate"` clean. |
| 15 | CI + docs | `.github/workflows/tests.yml`, `PIPELINE.md`, `ARCHITECTURE.md`, `README.md` | Sweep lists `shared.tokenize`, `shared.extract`, `shared.chunking`, `shared.figures`, `shared.ingest_v2`, `agents.agent8_verifier`; `ARCHITECTURE.md` gains §4.7–4.9 (text in the plan); `PIPELINE.md` Agent 3 row, layout tree and config notes updated; one sentence in `README.md`'s "What he's worst at". |

### Stage E — the live run (local only, gated — stop at the first failed gate)

| # | Step | Command / check | Gate |
|---|---|---|---|
| 16.1 | Baseline | `python scripts/index_stats.py` (script in the plan) | Prints `36,609 chunks over 323 docs`, `median 242`, `< 200 chars: 45%`. Keep the output. |
| 16.2 | **One paper, figures ON** | `CITATION_INDEX_VERSION=2 python -m research_assistant.agents.agent6_manual_ingestor --once data/pulled_pdfs/doi_10.1002_adma.202211157.pdf --citation "Covalent MoS2 networks" --describe-figures` (~6 min) | Log shows `figure analysis ON`, `grobid`, `11 caption(s), 11 described, 53 bib entries kept out`, `tokenizer v2`. `data/ingested_v2.json` and `data/bm25_index_v2.pkl` exist; `data/ingested.json` mtime unchanged. Open two crops in `data/images/` and read their descriptions out of `physics_papers_v2`: each must be about *that* figure. **If a description is about the wrong figure, stop.** |
| 16.3 | Full corpus, figures OFF | `CITATION_INDEX_VERSION=2 nohup python -m research_assistant.agents.agent3_ingestor --workers 1 > <log> 2>&1 &` (45–90 min) | Ends with `Processed 322, skipped 1`; a warning names any papers that fell back to PyMuPDF (the garbled-font one at least). |
| 16.4 | Compare | `CITATION_INDEX_VERSION=2 python scripts/index_stats.py`; `python scripts/index_stats.py`; `stat -c %y data/ingested.json data/bm25_index.pkl` | v2: roughly 8–14k chunks, median ~1,000–1,200, <5% under 200 chars, extraction mostly `grobid`. v1: identical to 16.1, files untouched. |
| 16.5 | The app on v2 | `CITATION_INDEX_VERSION=2 streamlit run app.py` | Tab 1 shows the toggle (don't build). Tab 2: cite one sentence → full-paragraph passages, no bibliography text. Tab 4: ask about the MoS₂ paper → answer cites it; any description source carries the "Auto-generated" marker. Tab 3: cite two sentences, Verify → verdicts, no `figure_description` text in evidence. Record each in the final commit message. |

## Accepted losses — do not "fix" these, they are decided

The human reviewed these and chose to proceed without them. Do not add work for them; do not be surprised by them.

1. **19 v1 papers have no PDF on disk** (the seed papers; `raw/` is empty). They will not be in v2. 16 have a cached TEI that a later change could ingest text-only.
2. **Table cell contents** are not indexed in v2 (caption only). GROBID's `<table>` rows are available if this is revisited.
3. **Appendices** (`<back><div type="annex">`) are not parsed. Body only.
4. **Display equations** (`<formula>`) are dropped from paragraph text.
5. **Crops of vector figures** may be caption-sized when GROBID emits no `<graphic>` (5 of 11 had one in the test paper). Gate 16.2 is where this is looked at; Detectron2 is not brought back in this plan.
6. **Research chat** injects `top_k=5` chunks per turn; at v2 chunk sizes this fills the 4k window faster. Not changed in this plan.
7. **No evaluation harness.** Task 16 is judged by eye. The harness is spec §2, deferred.

## Definition of done

- 16 commits on `ingestion-v2`, one per task, messages as in the plan.
- `CITATION_LOG_FILE=0 python -m pytest tests/ -v` green; CI's import-sweep loop passes locally with the new modules added.
- Task 16's five gates recorded (numbers and tab observations) in the last commit message.
- v1 files byte-identical to before (`ingested.json`, `bm25_index.pkl` mtimes; `physics_papers` count 36,609).
- Nothing pushed, nothing merged, nothing deployed. Hand back with the branch name and the Task 16 numbers.
