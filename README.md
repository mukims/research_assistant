# Research Assistant Pipeline

A multi-agent pipeline that builds a citable knowledge base from physics papers.
From a one-line research idea it finds a seed paper, walks its reference list,
fetches the open-access PDFs it can find, and ingests them — chunk-level into a
hybrid (vector + keyword) index and paper-level as a one-paragraph summary. It
then answers two questions: *what has already been done on this idea* (a
related-work synthesis) and *which source backs this sentence* (a LaTeX
citation) — for one sentence or for a whole draft at once.

Retrieval is two-stage: match the idea against the paper **summaries** first,
let the LLM drop the off-topic ones, then run the detailed chunk search only
over what survives — so it stays fast as the corpus grows.

How to drive the app is in [HOW_TO_USE.md](HOW_TO_USE.md) (also shown in the
app's *How to use* tab). Setting up GROBID, which Agent 1 needs for good
reference extraction, is in
[Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).
Design rationale for every major choice is in
[ARCHITECTURE.md](ARCHITECTURE.md).

Models run locally through [Ollama](https://ollama.com) by default, or against
any OpenAI-compatible API (`LLM_BACKEND=openai`). A Streamlit UI
([app.py](app.py)) and a [Dockerfile](Dockerfile) are included. The
bibliographic lookups (Agent 0: arXiv / OpenAlex / Semantic Scholar; Agent 2:
Crossref, Unpaywall, Europe PMC, arXiv) always go out to those services.

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| **Agent 0 — Discoverer** | [agent0_discoverer.py](research_assistant/agents/agent0_discoverer.py) | Takes a free-text research idea and searches relevance-ranked indexes in turn — arXiv, then OpenAlex, then Semantic Scholar — downloading the first result whose PDF actually fetches (paywalled publisher links are skipped, not fatal). Saves it into `data/raw/` under a `source_key`-derived name and records the query → paper in `seed_papers.json`. If nothing downloads, `--url` / `discover_from_url()` takes an arXiv or direct-PDF link instead, and `--seed-file` / `discover_from_file()` seeds from a PDF you already have — validating the magic bytes, reading a title / DOI / arXiv id out of it, and filing it under the same `source_key` scheme (falling back to a content hash). With that path the research query is optional: left blank, it is inferred from the paper's title or filename. From here the rest of the pipeline runs unchanged. |
| **Agent 1 — Extractor** | [agent1_extractor.py](research_assistant/agents/agent1_extractor.py) | Sends every PDF in `data/raw/` to a GROBID server and parses the TEI output into each paper's own metadata plus its full reference list (title, authors, year, DOI, raw string), scoring each consolidated DOI against the printed reference so grey-literature mismatches can be flagged. **When GROBID is unreachable — or returns no reference list for a particular paper — it falls back** to pattern-matching a numbered reference list (`[1]`, `1.`, `(1)`) out of that PDF's `pdftotext` output instead. The fallback is weaker (a raw string per reference, no DOIs, no authors/year), but it keeps the pipeline moving instead of dead-ending on a single external Java service. The path taken — `grobid`, `regex`, or `none` — is recorded per PDF, and a per-run count sits under `"extraction"` in `extracted_citations.json`. |
| **Agent 2 — Fetcher** | [agent2_fetcher.py](research_assistant/agents/agent2_fetcher.py) | Collapses the references to distinct sources, resolves a DOI per source (trusting Agent 1 when it was confident, otherwise asking Crossref), and tries to download an open-access PDF from Unpaywall → Europe PMC → arXiv. Writes `downloaded.json` / `failed_downloads.json` atomically after every paper, so a crashed run resumes where it stopped. A recorded failure is skipped next run only when it was definitive (paywalled, not indexed, 404); a transient one (dropped connection, rate limit, 5xx, truncated body) is retried. |
| **Agent 3 — Ingestor** | [agent3_ingestor.py](research_assistant/agents/agent3_ingestor.py) | For each downloaded PDF: semantic chunking of the text, embedding into ChromaDB, a rebuild of the BM25 index, and **one LLM summary per paper** into a separate `physics_summaries` collection. With layout detection on, it also crops figures/tables and keeps their captions (a VLM description of each is opt-in, `CITATION_FIGURE_VLM=1`). Tracks what's ingested in `data/ingested.json` so re-runs are cheap. |
| **Two-stage retrieval** | [shared/retrieve.py](research_assistant/shared/retrieve.py) | Stage 1: rank papers by summary similarity → LLM gate ("relevant prior work? Y/N") → shortlist. Stage 2: hybrid chunk search restricted to the shortlist. `research_answer()` then synthesises a related-work overview. |
| **Agent 4 — Assistant** | [agent4_assistant.py](research_assistant/agents/agent4_assistant.py) | Given a single sentence of draft text, runs hybrid search over the corpus and asks the LLM to rewrite the sentence with the correct `\cite{key}` inserted, plus an explanation of why that source supports the claim. |
| **Agent 5 — Batch citer** | [agent5_batch_citer.py](research_assistant/agents/agent5_batch_citer.py) | The same job as Agent 4, over a whole draft file at once. Splits the text into sentences, batches an LLM citation-need check across them, then cites every sentence that needs it and the corpus supports. Writes the cited draft, a `_citations.json` key → source mapping (BibTeX input), and a `_report.md` explaining every decision sentence by sentence. Aborts and writes nothing if a batched verdict list cannot be aligned back to its input sentences, rather than guessing which verdict belongs to which sentence. |
| **Agent 6 — Manual ingestor** | [agent6_manual_ingestor.py](research_assistant/agents/agent6_manual_ingestor.py) | Watches `data/pulled_pdfs/` for PDFs dropped in by hand — papers Agent 2 couldn't find automatically (e.g. paywalled, so downloaded manually) — and ingests each one through the same pipeline as Agent 3, with the same already-ingested check. `--once <file>` ingests a single PDF immediately instead of watching. |
| **Agent 7 — Research chat** | [agent7_research_chat.py](research_assistant/agents/agent7_research_chat.py) | A multi-turn conversational agent over the ingested corpus, with `/sources` and `/export` (to a timestamped markdown file) commands. Streams its answers token-by-token via `shared.llm.chat_stream()` instead of returning one block. |

Agent 0 just leaves a PDF in `data/raw/`, which is exactly what Agent 1 already
reads — nothing downstream needs to know it ran. The reference-list format
written by Agent 1 is consumed by Agent 2, whose manifest is consumed by
Agent 3, whose two indexes (chunks + summaries) are queried by the retrieval
layer and by Agents 4, 5 and 7.

**[orchestrate.py](orchestrate.py)** runs the whole chain for one research idea
as a **LangGraph** state machine:

```
discover ─(no seed, --ask)─► fallback ─► END      answer from the model alone
   │      ─(no seed)──────────────────► END
   ▼
ingest_seed ─► extract ─(both strategies found nothing)─► END
                 ▼
               fetch ─► ingest_refs ─(--ask)─► respond ─► END
                             └───────────────────────────► END
```

Each node wraps the same agent entry point you'd otherwise call by hand; the
conditional edges handle "no seed paper found" (answer from general knowledge,
still asking for a PDF link) and "neither extraction strategy found anything to
fetch". A dead GROBID server no longer ends the run on its own — Agent 1
degrades to the regex fallback instead (see the Agent 1 row above), so this
branch only fires when *that* also comes up empty.
Every agent keeps its own on-disk state, so a run that dies partway is resumed
by running it again — finished stages no-op.

```bash
python orchestrate.py --query "topological protection in disordered quantum wires"
python orchestrate.py --query "..." --ask --workers 4
```

**[watch.py](watch.py)** is the other entry point: instead of one query-driven
run, it watches `data/raw/`, `data/pulled_pdfs/` and `data/drafts/` and runs
the matching agents whenever a file appears in one of them (debounced per
directory), so you can leave it running and just drop files where they
belong.

```bash
python watch.py               # watch mode
python watch.py --chat        # watch mode + an interactive research chat in the foreground
```

## Layout

```
research_assistant/                the checkout root
├── pyproject.toml                 editable install; package + pytest config
├── requirements.txt               core (UI, orchestration, text-only ingestion)
├── requirements-layout.txt        optional: detectron2 + torch layout stack
├── requirements-test.txt          the light packages CI installs
├── Dockerfile  .dockerignore  .gitignore
├── publaynet_config.yaml
├── model_final.pth                Detectron2 / PubLayNet checkpoint (optional)
│
├── app.py                         Streamlit UI (also the container entry point)
├── orchestrate.py                 LangGraph — one research idea, end to end
├── watch.py                       watchdog daemon — directory-driven
│
├── research_assistant/            the installable package
│   ├── config.py                  every model name, path and tunable
│   ├── prompts.py                 every prompt sent to a model
│   ├── schemas.py                 inter-stage contracts: Reference, DownloadedPaper, SeedPaper
│   ├── agents/
│   │   ├── agent0_discoverer.py       idea → seed paper
│   │   ├── agent1_extractor.py        PDF → reference list (GROBID + regex fallback)
│   │   ├── agent2_fetcher.py          references → open-access PDFs
│   │   ├── agent3_ingestor.py         PDFs → chunks + summaries
│   │   ├── agent4_assistant.py        one sentence → citation
│   │   ├── agent5_batch_citer.py      whole draft → cited draft + report
│   │   ├── agent6_manual_ingestor.py  dropped file → corpus
│   │   └── agent7_research_chat.py    multi-turn chat over the corpus
│   └── shared/
│       ├── llm.py          backend-agnostic chat / chat_stream / embeddings
│       ├── ingestion.py    process → upsert → mark → index
│       ├── manifest.py     what has been ingested (data/ingested.json)
│       ├── search.py       hybrid BM25 + dense with Reciprocal Rank Fusion
│       ├── retrieve.py     two-stage retrieval (summary shortlist → deep chunk search)
│       ├── fetch.py        stream-a-PDF-to-disk-with-validation
│       ├── source_key.py   deterministic identity for a reference / document
│       ├── atomic.py       write-temp-then-replace, for every manifest on disk
│       └── db.py  log.py  retry.py
│
├── tests/                          pytest suite (collects unittest.TestCase classes unchanged)
├── .github/workflows/tests.yml
└── data/                           all runtime state (gitignored; CITATION_DATA_DIR overrides)
    ├── raw/                        PDFs to process; raw/grobid_output/ caches GROBID's TEI
    ├── pulled_pdfs/                PDFs from Agent 2, or dropped by hand for Agent 6
    ├── drafts/                     drop a .txt here and watch.py runs Agent 5 on it
    ├── images/                     figure/table crops from ingestion (debug artefact)
    ├── logs/
    ├── physics_vectordb/           persistent ChromaDB store
    ├── seed_papers.json  extracted_citations.json  downloaded.json
    ├── failed_downloads.json  ingested.json  bm25_index.pkl
    └── ingest.lock                 held while a batch is ingesting
```

Three runnable entry points sit at the root because they are the things a user
actually invokes — `app.py`, `orchestrate.py`, `watch.py`; everything they
import lives in the `research_assistant` package, installed editable so it
resolves regardless of the working directory the scripts are run from.

Everything the pipeline **writes** now lives under `data/` (`CITATION_DATA_DIR`,
default `<project>/data`, no longer the repo root) — set it to a writable path
such as `/data` on a read-only or ephemeral host.

## Prerequisites

> **Not a developer?** [HOW_TO_USE.md](HOW_TO_USE.md) opens with a
> step-by-step setup walkthrough — every command verified on a clean machine,
> with a table of what each failure message means. Start there instead.

- Python 3.10–3.12 (CI runs 3.10 and 3.12; **not 3.13+** — `lxml==4.9.4` has
  no 3.13 wheel and fails to build from source against it):
  `pip install -e .` then `pip install -r requirements.txt`.
- **An LLM + embedding backend** (`LLM_BACKEND`, `EMBED_BACKEND`):
  - `ollama` (default) — a local [Ollama](https://ollama.com) daemon with the
    models in `config.py` pulled.

    Note that `LLM_MODEL` defaults to `gemma4:31b-cloud`, which runs on
    Ollama's servers rather than yours: it needs `ollama signin` and sends
    text off the machine. For a fully local setup, pull `gemma4:e2b-mlx` and
    `nomic-embed-text` and set `CITATION_LLM_MODEL=gemma4:e2b-mlx`.
  - `openai` — any OpenAI-compatible endpoint (`OPENAI_BASE_URL`,
    `OPENAI_API_KEY`); `huggingface` for embeddings via `huggingface_hub`.
- **GROBID** for Agent 1 — `GROBID_SERVER` defaults to
  `http://localhost:8070` (`curl localhost:8070/api/isalive`). The expected
  setup is a local container:
  `docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1` —
  [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md)
  covers installing Docker and troubleshooting it. You can point the env var
  at a hosted instance instead.

  A server that is down, or that returns no reference list for a paper, does
  not stop the pipeline — Agent 1 falls back to the weaker regex extraction
  for that paper (see the Agent 1 row above). Note that this degrades
  **silently**: references then carry no DOIs, authors or years, and Agent 2
  fetches noticeably less. Check the GROBID indicator before concluding the
  pipeline is performing badly.
- **Layout detection is optional** (`CITATION_LAYOUT_DETECTION=1`). It needs
  `pip install -r requirements-layout.txt` plus detectron2 from source, and a
  torch build. With it off, ingestion is text-only (PyMuPDF).

## Usage

Install the package once, editable, so `research_assistant.*` resolves from
anywhere and the three root scripts can import it:

```bash
pip install -e .
pip install -r requirements.txt
```

The whole pipeline for one idea:

```bash
python orchestrate.py --query "topological protection in disordered quantum wires" --ask
```

Or run the stages by hand:

```bash
# 0. Seed from a research idea — finds and downloads a relevant paper into data/raw/
python -m research_assistant.agents.agent0_discoverer --query "topological protection in disordered quantum wires"
#    no open-access hit? give it a link:
python -m research_assistant.agents.agent0_discoverer --query "..." --url https://arxiv.org/abs/2401.12345
#    or seed from a PDF you already have — query optional, inferred from the paper:
python orchestrate.py --seed-file path/to/paper.pdf --ask
#    (or skip Agent 0 and drop your own PDF(s) into data/raw/ by hand)

# 1. Mine the reference list of everything in data/raw/
python -m research_assistant.agents.agent1_extractor        # -> extracted_citations.json

# 2. Download the open-access PDFs of those references
export UNPAYWALL_EMAIL="you@example.com"   # Unpaywall requires a contact address
python -m research_assistant.agents.agent2_fetcher          # -> pulled_pdfs/, downloaded.json, failed_downloads.json

# 3. Ingest the downloaded PDFs into the search index
python -m research_assistant.agents.agent3_ingestor         # -> physics_vectordb/, bm25_index.pkl
python -m research_assistant.agents.agent3_ingestor --workers 4 --force   # parallel parse, re-ingest everything

# 4. Ask for a citation for one sentence of draft text
python -m research_assistant.agents.agent4_assistant --text "Anderson localization suppresses diffusion in 1D." --top_k 3

# 5. Cite a whole draft
python -m research_assistant.agents.agent5_batch_citer --file draft.txt --out cited.txt

# 6. Ingest a PDF you downloaded by hand (Agent 2 couldn't reach it)
python -m research_assistant.agents.agent6_manual_ingestor --once path/to/paper.pdf

# 7. Chat with the corpus
python -m research_assistant.agents.agent7_research_chat

# Or run the reactive daemon instead of the per-query pipeline
python watch.py
```

Or the Streamlit UI (four interactive agents across five tabs — the fifth
tab is documentation, not an agent — with a browser front-end):

```bash
streamlit run app.py
```

## Configuration notes

- `config.py` is the single place for model names, directory paths, and every
  tunable (chunking thresholds, rate limits, RRF constant, batch sizes).
- Writes are anchored to `CITATION_DATA_DIR` (default: `<project>/data`);
  code is anchored to the installed package, so scripts can be run from
  anywhere.
- Backend: `LLM_BACKEND`, `EMBED_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` /
  `HF_TOKEN`, `CITATION_LLM_MODEL`, `CITATION_CHAT_MODEL`, `CITATION_EMBED_MODEL`.
- Ingestion: `CITATION_LAYOUT_DETECTION`, `CITATION_FIGURE_VLM`,
  `CITATION_DETECTRON_WEIGHTS`, `CITATION_DETECTRON_CONFIG`, `CITATION_IMAGES_DIR`,
  `CITATION_SUMMARY_MODEL`, `CITATION_SUMMARY_MAX_CHARS`.
- Retrieval: `CITATION_DOC_SELECT_K` (stage-1 shortlist size),
  `CITATION_DOC_GATE` (LLM relevance gate, default on).
- Agent 1: `GROBID_SERVER`, `GROBID_BATCH_CONCURRENCY`.
- Agent 0: `CITATION_SEARCH_PROVIDERS` (default `arxiv,openalex,semanticscholar`,
  tried in order), `OPENALEX_MAILTO`, `S2_API_KEY` (optional Semantic Scholar
  key — the keyless pool is heavily rate-limited).
- Misc: `UNPAYWALL_EMAIL`, `CITATION_LOG_DIR`, `CITATION_LOG_FILE=0`.

## Development

```bash
pip install -e .
pip install -r requirements-test.txt
CITATION_LOG_FILE=0 python -m pytest tests/ -v
```

The suite exercises pure logic — citation parsing, paper naming, extractor
filing, ingestion bookkeeping, retrieval ranking — and needs none of the heavy
stack (ChromaDB, PyTorch, detectron2, an LLM backend, GROBID). CI
(`.github/workflows/tests.yml`) runs it on Python 3.10 and 3.12 and separately
import-sweeps every module in the package to catch a broken import that no
test happens to cover.
