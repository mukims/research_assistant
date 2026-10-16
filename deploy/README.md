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
