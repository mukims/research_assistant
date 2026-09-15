# Deploying to Google Cloud

One VM, two containers under Docker Compose, one firewall rule. See
`notimportant/superpowers/specs/2026-09-09-gcp-deployment-design.md` for architectural design rationale.

## Architecture

* **VM**: `e2-custom-4-12288` (4 vCPU, 12 GB RAM) on Debian 12
* **Storage**: 50 GB `pd-balanced` persistent disk mounted at `/mnt/disks/data`
* **Containers**:
  * `app`: Streamlit research assistant UI and Agent 0–8 pipeline (port 8080)
* **Backend**:
  * **Default (CPU VM / Local)**: Ollama for chat, synthesis & judgement (`gemma4:e2b`, `num_ctx: 4096`) and embeddings (`nomic-embed-text`).
  * **Cloud Hosted**: Google Gemini (`gemini-3.5-flash-lite` via OpenAI-compatible endpoint) when `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set; embeddings remain on Ollama (`nomic-embed-text`) so existing vector indexes stay consistent.
* **Secrets**: `GEMINI_API_KEY` (or `OPENAI_API_KEY` / `HF_TOKEN`) stored in Google Secret Manager, pulled into `/etc/app.env` (or persistent `/mnt/disks/data/.env`) at VM boot.

---

## Quick Start

### 1. First-Time Setup

```bash
./deploy/00-prereqs.sh   # Enable APIs, create Artifact Registry repo, store OpenAI API key
./deploy/10-vm.sh        # Create service account, persistent disk, VM, and firewall rule
./deploy/deploy.sh       # Build image, push to Artifact Registry, and launch containers on VM
```

### 2. Updating / Redeploying Code

```bash
./deploy/deploy.sh
```

### 3. If the App is Unreachable

Access is restricted to a single `/32` IP address. If you move networks (venue Wi-Fi, home, mobile tether):

```bash
./deploy/allow-ip.sh
```

**Alternative (Zero-Firewall Encrypted Access)**:
If public port 8080 is blocked by venue Wi-Fi or you want complete privacy:
```bash
gcloud compute ssh research-assistant --zone europe-west1-b -- -L 8080:localhost:8080
```
Then navigate to `http://localhost:8080` in your browser.

---

## The standalone image (what the VM actually runs)

`deploy.sh` builds `Dockerfile` (app only, OpenAI backend) and pushes it to
Artifact Registry. The container that serves
`https://marvin-the-citebot.duckdns.org/` is **not** that image: it is
`Dockerfile.standalone` — Ollama with pre-baked `gemma4:e2b` +
`nomic-embed-text`, GROBID, and the app in one container, started by
`deploy/entrypoint.sh` (Ollama → GROBID → Streamlit, each waited for). It is
built on the VM itself; its build context needs two things a fresh clone does
not have:

| input | what it is | how to make it |
|---|---|---|
| `models/` | an Ollama models directory (~7 GB), copied to `/opt/ollama/models` so the image never downloads at start-up | `OLLAMA_MODELS=$PWD/models ollama pull gemma4:e2b` and the same for `nomic-embed-text` |
| `/mnt/disks/data/.env` | run-time env on the data disk; overrides the image's `ENV` — this is where `LLM_BACKEND=openai` + `GEMINI_API_KEY` switch the judge to Gemini while embeddings stay on in-container Ollama | written once by hand; never in the image |

The container is not compose-managed (the tracked `docker-compose.yml` is the
registry path's; the `research-assistant-grobid-1` sidecar is the only thing
it runs on the VM). It was started by hand, and this is the exact shape of
that container today — `docker inspect research-assistant` is the authority:

```bash
# on the VM, from a synced checkout that also contains models/
docker build -f Dockerfile.standalone \
  -t europe-west1-docker.pkg.dev/researchassistant-508111/research-assistant/standalone:latest .
docker run -d --name research-assistant --restart always \
  -p 8080:8080 -v /mnt/disks/data:/home/user/data \
  europe-west1-docker.pkg.dev/researchassistant-508111/research-assistant/standalone:latest
```

`--restart always` survives a VM reboot but **not** a manual `docker stop` —
after one, only `docker start research-assistant` brings the site back.

**Hot-patching code without a rebuild** (what a routine code deploy looks like
today — a rebuild re-copies 7 GB of models):

```bash
tar czf /tmp/patch.tgz app.py research_assistant scripts
gcloud compute scp /tmp/patch.tgz research-assistant:/tmp/patch.tgz --zone europe-west1-b
gcloud compute ssh research-assistant --zone europe-west1-b --command 'bash -s' <<'EOF'
  mkdir -p /tmp/patch && tar xzf /tmp/patch.tgz -C /tmp/patch
  for f in /tmp/patch/*; do docker cp "$f" research-assistant:/home/user/app/; done
  docker exec research-assistant chown -R 1000:1000 /home/user/app
  docker restart research-assistant
EOF
```

The patched files live in the container's writable layer: `docker restart`
keeps them; `docker rm` + `docker run`, or a new image, does not.
Verify after a restart with
`curl -s localhost:8080/_stcore/health` and
`docker exec research-assistant curl -s localhost:11434/api/tags` — an audit
started before Ollama answers is aborted by the breaker (and, since
`0870338`, not cached).

---

## Controlling Costs

* **Running**: ~$95/mo (VM compute ~$89 + 50 GB disk ~$6).
* **Stopped**: ~$6/mo (Compute billing stops completely; disk persists).

Use the control helper:
```bash
./deploy/vm-control.sh stop     # Pause compute billing
./deploy/vm-control.sh start    # Resume VM and update firewall rule
./deploy/vm-control.sh status   # Check VM status and IP
```

---

## Pre-Demo Checklist

1. **Start the VM 10 minutes early**: GROBID's JVM takes 30–60s to initialize and the first request warms caches.
2. **Run `./deploy/allow-ip.sh` from the venue Wi-Fi**: Ensures your current public IP is authorized.
3. **Confirm the corpus is seeded**: A fresh VM starts with an empty corpus; ensure papers have been indexed.
4. **Warm the pipeline**: Run one search and citation verification prior to presenting.
5. **Do not run batch ingestion during a live presentation**: Ingestion engages GROBID with multiple parallel extraction workers.
