# Running Marvin

## On this machine

```bash
deploy/local.sh up              # start GROBID if it is not answering, then the app
deploy/local.sh up --no-grobid  # app only (see "low memory" below)
deploy/local.sh status          # what is answering; changes nothing
deploy/local.sh down            # stop the GROBID container
```

* **This machine only.** The app listens on `127.0.0.1`. Streamlit's own default
  is every network interface, which is what `streamlit run app.py` gives you
  without the flag.
* **GROBID** is started under the container name and image the app's own
  Start/Stop buttons use (`GROBID_CONTAINER_NAME`, default `grobid`;
  `GROBID_DOCKER_IMAGE`, default `grobid/grobid:0.8.1`), so the two never fight.
  An existing container of that name is started, not recreated, whatever image it
  was built from. A `GROBID_SERVER` that is not on this machine is left alone.
  Published on `127.0.0.1:8070` only.
* **Nothing else is touched.** No other container is started, stopped or removed.
* **Degrades, does not block.** No Docker, a container that dies on start, or a
  GROBID that never answers: the app still starts, and the warning says what is
  lost (reference lists fall back to regex extraction: no DOIs, no authors,
  fewer downloads).
* **Ollama** is checked, not managed. It is reported as up with its model count,
  or as not reachable, which is fine if you use a hosted backend.
* **Settings:** `PYTHON` (the interpreter that has the requirements; default
  `python3`), `PORT` (default 8501), `GROBID_SERVER`, `GROBID_WAIT` (seconds to
  wait for GROBID, default 120). Everything else — models, backends, the data
  directory — is read by the app from its usual environment and `.env`.
* **Low memory:** GROBID is a Java server. If Docker's memory limit is too small
  the kernel kills it, and `docker ps -a` shows `Exited (137)`. Raise Docker
  Desktop's memory limit, or use `--no-grobid`.

`tests/test_deploy_scripts.py` runs `local.sh` under bash against stubbed `docker`,
`curl` and Python; nothing real is started or removed.

## Rebuilding or Migrating to Index v3 (Docling & LanceDB)

> **Note:** The legacy `scripts/migrate_to_v3.py` and `scripts/export_old_labels.py` helpers from the early Chroma-to-v3 transition have been retired; index v3 now runs on **LanceDB** + **Docling** via `scripts/rebuild_corpus_docling.py`. `deploy/migrate_setup.sh` remains available as a machine pre-flight/setup helper (Python 3.10–3.12 virtualenv, Ollama `embeddinggemma-2` pull, and Docker GROBID startup; pass `--check-only` to inspect without modifying anything).

To rebuild or refresh the corpus into LanceDB with multi-modal Docling extraction, clean figure crops, and zero-day MVCC table compaction:

```bash
# Optional dry run to check candidate counts and OCR preflight
python scripts/rebuild_corpus_docling.py --dry-run

# Run full extraction and ingestion into LanceDB (with --no-ocr to prevent memory spikes)
python scripts/rebuild_corpus_docling.py --batch-size 10 --no-ocr
```

- **Resumes automatically**: Uses `data/ingested_docling.json` (or `data/ingested_v3.json`) to skip already-ingested papers.
- **Deduplication**: Reconciles files against `data/corpus_dedupe.json`, ignoring quarantined and alias duplicates.
- **Compaction**: Automatically executes zero-day table compaction (`cleanup_older_than=timedelta(seconds=0)`) every 10 batches and upon completion, keeping LanceDB disk usage compact.
- **Tantivy FTS**: Keeps full-text search indexes on `text` and `citation_source` intact across ingest batches and refreshes them upon completion.
- **Same model build everywhere**: `embeddinggemma-2` embeddings are tied to the index version; do not override `CITATION_EMBED_MODEL` in `.env` or `data/.env` when opening an existing v3 store.

