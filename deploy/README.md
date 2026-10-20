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

## Migrating to index v3 on another machine

The migration (`scripts/migrate_to_v3.py`) re-ingests every paper the v1 and v2 manifests name. It
runs anywhere with Python 3.10-3.12, Ollama 0.36+, Docker (for GROBID) and a summary model; it reads
your old data and writes only the new index. The time is an estimate, not a measurement: about 50 s per paper with a fresh GROBID call in trials, much less where a usable
GROBID output is reused, so plan on several hours for 874 papers. It resumes where it stopped.

**1. Get the code.** Check out the `feat/v3-gemma-images` branch (or the merged result).

**2. Copy the old data, inputs only** (2.46 GB; no Chroma store, no `.env`):

```bash
rsync -a --prune-empty-dirs \
  --include='/ingested.json' --include='/ingested_v2.json' --include='/downloaded.json' \
  --include='/pulled_pdfs/***' --include='/raw/***' --include='/raw_pdfs/***' \
  --exclude='*' data/ USER@HOST:migration-source/
```

`downloaded.json` carries each paper's citation label (the `\cite{}` key); without it papers are
labelled by file name. `raw/grobid_output/` lets the migration reuse GROBID output that has figure
boxes and re-read the rest.

**3. Prepare the machine** (installs and starts, never deletes; `--check-only` shows what is missing):

```bash
deploy/migrate_setup.sh --check-only
deploy/migrate_setup.sh                       # venv + requirements, Ollama model, GROBID on 127.0.0.1:8070
source .venv-migrate/bin/activate
```

It does not choose a summary model for you: put `GEMINI_API_KEY=...` in `.env`, or `ollama pull gemma4:e2b`.
Do not set `CITATION_EMBED_MODEL` in `.env` or `<data>/.env`; it overrides the `embeddinggemma-2` default
and the migration refuses to start.

**4. Run it**, with the new index written under `CITATION_DATA_DIR`:

```bash
export CITATION_DATA_DIR=$HOME/marvin-data PYTHONPATH=.
python scripts/migrate_to_v3.py --from-data-dir ~/migration-source --dry-run   # counts, labels, GROBID cache
python scripts/migrate_to_v3.py --from-data-dir ~/migration-source --limit 3   # trial
python scripts/migrate_to_v3.py --from-data-dir ~/migration-source            # everything
python scripts/migrate_to_v3.py --from-data-dir ~/migration-source --verify    # complete?
```

It stops, and says why, if GROBID stops answering or a batch falls back to PyMuPDF (those papers would
have no figures yet be marked done). Then use `CITATION_DATA_DIR` the same way for the app.

**Keep the figure crops** (`<data>/images/`): a future embedder re-embeds them from there.

**Same model build everywhere.** `embeddinggemma-2:latest` is a moving name. A new index records the
build digest of the machine that made it and warns when another machine's build differs, because
vectors from one build queried with another retrieve slightly worse without any error. The setup script
prints the digest; pull the same build on any machine that will query this index.

`tests/test_deploy_scripts.py` also runs `migrate_setup.sh` under stubs (a copy in a throwaway project root).

