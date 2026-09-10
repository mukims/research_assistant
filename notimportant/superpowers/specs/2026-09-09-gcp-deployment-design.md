# GCP deployment — GCE VM, Compose, firewalled

*2026-09-09*

## 1. Purpose

Deploy the research assistant — including Agent 8, merged at `239328b` — to
Google Cloud as a single-user, always-reachable instance with a persistent
corpus.

`ARCHITECTURE.md` §8 already describes a Cloud Run target. **That section is
wrong and this spec corrects it** (§4.1). No deployment artefacts exist in the
repo today: no `docker-compose.yml`, no `.gcloudignore`, no deploy scripts.

### The constraint that shapes this deployment

**It has to work during a live demo.** That is the stated purpose, and it drives
three choices that would otherwise go the other way: 4 vCPU so GROBID cannot
freeze the UI mid-extraction (§4.5), a pre-demo warm-up and smoke check rather
than trusting a cold VM (§8), and a one-command firewall update so presenting
from an unfamiliar network is not a crisis (§4.2).

No domain is available, so IAP and managed certificates are out — see §4.2.

## 2. What is being deployed

One VM running two containers under Docker Compose:

| Container | Image | Role |
|---|---|---|
| `app` | built from this repo's `Dockerfile` | Streamlit UI, all eight agents |
| `grobid` | `grobid/grobid:0.8.1` | reference extraction for Agent 1 |

Chat and embeddings go to OpenAI. The corpus lives on a persistent disk mounted
at `CITATION_DATA_DIR`.

## 3. Layout

```text
docker-compose.yml            # app + grobid, PD bind-mount, restart policy
.gcloudignore                 # keep data/ and model_final.pth out of uploads
.env.example                  # every env var the deployment reads
deploy/
├── config.sh                 # shared names/region, sourced by the others
├── 00-prereqs.sh             # enable APIs, create Artifact Registry + secret
├── 10-vm.sh                  # VM, persistent disk, service account, firewall
├── allow-ip.sh               # repoint the firewall at your current public IP
├── deploy.sh                 # build → push → pull → compose up on the VM
├── startup.sh                # VM startup script: mount PD, docker login, up
└── README.md                 # first-run, redeploy, and pre-demo checklist
```

Changed: `Dockerfile`, `.dockerignore`, `pyproject.toml`, `ARCHITECTURE.md`,
`HOW_TO_USE.md`.

## 4. Structural decisions

### 4.1 GCE VM, not Cloud Run

`ARCHITECTURE.md` §8.2 says the corpus survives if "a GCS volume is mounted at
`CITATION_DATA_DIR`". It does not.

The corpus is `chroma.sqlite3` plus mmap'd HNSW `.bin` files — 1.2 GB today.
Cloud Storage FUSE has no POSIX file locking and turns small random writes into
whole-object rewrites. SQLite and a memory-mapped vector index on that substrate
is data corruption, not slow-but-correct.

A GCE VM with a persistent disk is an ordinary filesystem, so ChromaDB, the BM25
pickle and the ingestion lock all work untouched. **No application code changes
are needed for persistence.**

**Trade-off.** Always-on cost instead of scale-to-zero, and the VM is a machine
to patch. Accepted: a scale-to-zero design that corrupts its corpus is not
cheaper, it is broken.

**Rejected alternatives.** Filestore (real NFS locking) starts at 1 TiB ≈
$200/mo. Cloud SQL + pgvector removes Chroma but rewrites `db.py`,
`retrieve.py` and `ingestion.py`, and leaves the BM25 pickle homeless.

### 4.2 Firewall to a single source IP, over HTTP

The app has no authentication and now spends an OpenAI key on every
verification run, so it must not sit open on a public IP. With no domain
available, IAP and Google-managed certificates are unavailable (managed certs
cannot be issued for a bare IP, and IAP's consent screen requires an authorised
domain).

A GCE firewall rule allowing `tcp:8080` from one `/32` source range is free,
takes one command, and is genuinely sufficient for a single user: an
unauthenticated app nobody can route to is not an exposed app.

**The demo hazard, and the mitigation.** Pinning to one IP means the app is
unreachable from any other network — conference wifi, a hotel, tethering. That
is precisely the failure mode a live demo invites. `deploy/allow-ip.sh` therefore
takes no arguments, discovers the caller's current public IP, and rewrites the
rule's source range in one command. Run it on arrival at the venue and access is
restored in seconds.

**Trade-off, stated plainly.** Traffic is plain HTTP, so the OpenAI key's
*effects* are protected by the firewall but the session itself is unencrypted
over the wire. For a single-user research tool reachable only from one address
this is acceptable; it would not be for anything multi-tenant or public.

**If a domain appears later**, §10 records what changes: IAP + a managed cert on
an HTTPS load balancer, ~$18/mo, and no other part of this spec moves.

### 4.3 GROBID runs under Compose; its lifecycle buttons go inert

`GROBID_SERVER=http://grobid:8070`, with `restart: unless-stopped` keeping the
container alive.

`shared/grobid_manager.py` drives `docker` commands to start and stop GROBID.
The app container has no Docker access, so Start / Stop / Restart will report
failure. The status probe — an HTTP call to `/api/isalive` — keeps working, so
the UI still shows whether GROBID is up.

**Why not mount the Docker socket.** It grants the app root-equivalent control
of the VM. That is a large security trade for a convenience the restart policy
already provides.

**No code change.** The buttons failing is acceptable; hiding them would mean
editing `app.py` for a cosmetic gain.

### 4.4 OpenAI for chat and embeddings; the corpus must be rebuilt

```
LLM_BACKEND=openai
CITATION_EMBED_BACKEND=openai
OPENAI_BASE_URL=https://api.openai.com/v1
CITATION_LLM_MODEL=gpt-4.1-mini
CITATION_EMBED_MODEL=text-embedding-3-small
```

No application code changes: `_openai_chat` and `_OpenAIEmbeddings` already
exist and are model-configurable.

**The existing corpus cannot be copied to the VM.** Its Chroma collection is
`dimension=768` (nomic-embed-text); `text-embedding-3-small` is 1536-dim, and
Chroma rejects the mismatch. Even forcing matching dimensions would be wrong —
vectors from two models do not share a space. The PDFs in `data/raw/` and
`data/pulled_pdfs/` are reusable; only the embeddings must be regenerated.
Re-ingest is a one-time operational step, well under a dollar.

**No Ollama runtime on the VM.** Every `ollama` / `langchain_ollama` import in
`shared/llm.py` is function-local and guarded by a backend branch, so with both
backends set to `openai` neither package is ever imported — verified by loading
every entry point under those env vars and confirming both are absent from
`sys.modules`. No daemon, no model pulls, no GPU. Both packages stay in
`requirements.txt` regardless, so the one image still runs the local Ollama path
per §8.1's "test what you deploy" — and they cost almost nothing, since
`langchain-ollama` rides on the `langchain-core` that `langchain-experimental`
already requires for the semantic chunker.

`gpt-4.1-mini` accepts `temperature=0`, which Agent 8 sends unconditionally.
A reasoning-tier model would reject it and turn every citation into an HTTP 400
reported as "Unusable model reply" — see §10.

### 4.5 e2-custom-4-12288 — four vCPU, twelve GB

**Memory.** GROBID's JVM is capped at 3 GB heap and reaches ~4 GB resident; the
app needs ~2–3 GB because `shared/db.py`'s `load_search_resources()` paginates
the entire collection into Python lists and builds the BM25 index in memory —
all 36,931 chunks resident, growing with the corpus. Plus ~0.5 GB for the OS.
At 8 GB that leaves ~0.5 GB spare; at 12 GB, ~4.5 GB, which is room to roughly
double the corpus.

**vCPU is the binding constraint, not memory.** Agent 1 drives GROBID at
`GROBID_BATCH_CONCURRENCY = 2`, and GROBID will use both cores of a 2-vCPU
machine during extraction — starving the Streamlit UI exactly while someone is
watching a batch run. Four vCPU lets GROBID take two and leaves two for the app.

**Cost.** ~$89/mo for the VM plus ~$6/mo for the 50 GB pd-balanced disk,
≈ **$95/mo**. Stopping the VM when idle zeroes the compute; the disk keeps
billing. Resizing later is stop → `set-machine-type` → start, with no data loss.

## 5. Components

### 5.1 `docker-compose.yml`

Two services. `app` builds from the repo image, reads its secrets from the
environment, publishes 8080, depends on `grobid`. `grobid` uses the upstream
image with `JAVA_OPTS=-Xmx3g` to bound the JVM. Both `restart: unless-stopped`.
The persistent disk is bind-mounted into `app` at `/home/user/data`
(= `CITATION_DATA_DIR`); `grobid` needs no volume.

`OPENAI_API_KEY` is read from the VM's environment, populated at boot from
Secret Manager by `startup.sh` — never baked into the image or the compose file.

### 5.2 `Dockerfile` changes

Swap the baked-in HF-router defaults for the OpenAI ones in §4.4. Everything
else stays: `$PORT`, `CITATION_LOG_FILE=0`, uid 1000,
`CITATION_LAYOUT_DETECTION=0`, `CITATION_DATA_DIR=/home/user/data`.

### 5.3 `pyproject.toml` — declare the judgement package data

```toml
[tool.setuptools.package-data]
"research_assistant.judgement" = ["prompt.md", "cases/*.jsonl"]
```

Carried forward from the judgement branch's final review. `judge.py` reads
`prompt.md` at **module** scope and `app.py` imports the verifier at module
scope, so under a non-editable install the Streamlit app fails to *start*, not
merely Agent 8. The image uses `pip install -e .` today, which masks it; this
declaration removes the trap before someone changes that line. Note `cases/`
has no `__init__.py`, so `packages.find` will never pick it up — `package-data`
is the correct mechanism, not a new package entry.

### 5.4 `.gcloudignore` and `.dockerignore`

`.gcloudignore` keeps `data/` (1.2 GB) and `model_final.pth` (856 MB) out of
build uploads; without it every deploy ships ~2 GB instead of ~2.6 MB.
`.dockerignore` currently lists stale top-level paths (`raw/*.pdf`,
`seed_papers.json`) that no longer exist now everything is under `data/` —
tighten it to match reality.

### 5.5 `deploy/` scripts

Numbered so the order is obvious, each idempotent and safe to re-run.

- **`00-prereqs.sh`** — enable the `compute`, `artifactregistry` and
  `secretmanager` APIs; create the Artifact Registry repo; create the
  `openai-api-key` secret, **prompting for the value** rather than taking it as
  an argument (a secret in `argv` lands in shell history and in every process
  listing on the machine).
- **`10-vm.sh`** — service account holding only `secretmanager.secretAccessor`
  and `artifactregistry.reader`; 50 GB pd-balanced disk; `e2-custom-4-12288`
  with `startup.sh` as metadata; firewall rule allowing `tcp:8080` from the
  caller's current `/32` only. Prints the VM's external IP.
- **`allow-ip.sh`** — takes no arguments. Discovers the caller's current public
  IP and rewrites the firewall rule's source range to match. This is the script
  to run on arrival at a demo venue; it is separate from `10-vm.sh` precisely so
  it can be run repeatedly without touching anything else.
- **`deploy.sh`** — build the image, push to Artifact Registry, then pull and
  `docker compose up -d` on the VM over SSH.
- **`startup.sh`** — VM metadata startup script: format the persistent disk on
  first boot only (guarded by `blkid`), mount it, `chown` to uid 1000 to match
  the image's non-root user, read the OpenAI key from Secret Manager into
  `/etc/app.env`, and `docker compose up -d`.

## 6. Boot sequence

```text
VM boots
  └─ startup.sh
       ├─ mkfs.ext4 the PD  (first boot only — guarded by blkid)
       ├─ mount /dev/disk/by-id/... → /mnt/disks/data
       ├─ chown 1000:1000   (matches the image's non-root user)
       ├─ read openai-api-key from Secret Manager → /etc/app.env
       └─ docker compose up -d
             ├─ grobid  :8070  (JVM warm-up ~40s)
             └─ app     :8080  → firewall (your /32) → you
```

## 7. Operations and failure handling

| Situation | Behaviour |
|---|---|
| VM reboots | `startup.sh` re-runs; PD already formatted, so it mounts and comes up |
| GROBID crashes | Compose restarts it; Agent 1 degrades to its regex fallback meanwhile |
| App crashes | Compose restarts it; the corpus is on the PD, so nothing is lost |
| Corpus corrupted | Rebuild from `data/raw/` — the PDFs are the source of truth |
| Cost runaway | The OpenAI key is the only metered resource; a spend limit belongs on the OpenAI account, not in this repo |

Stopping the VM when idle drops compute to zero; the 50 GB persistent disk
continues to bill at ~$6/mo. There is no load balancer to keep running.

## 8. Verification, and the pre-demo checklist

The deployment is correct when, in order:

1. `docker compose ps` on the VM shows both services healthy.
2. `curl -f http://localhost:8080/_stcore/health` succeeds on the VM.
3. The app loads in a browser from the allowed IP, and is refused from any
   other network (verify by phone tether — an untested firewall rule is an
   assumption, not a control).
4. Tab 1 seeds a paper, Tab 3 cites a draft, and **Verify citations** returns a
   verdict table. This one flow exercises OpenAI chat, OpenAI embeddings,
   ChromaDB on the persistent disk, and Agent 8 end to end.
5. `gcloud compute instances stop` then `start`, and the corpus survives.

### Pre-demo checklist

Because §1 names a live demo as the constraint, these are requirements rather
than suggestions:

- **Start the VM at least ten minutes ahead.** GROBID's JVM takes 30–60 s to
  become responsive and the first request after boot is the slowest.
- **Run `deploy/allow-ip.sh` from the venue's network** before you need the app.
  This is the single most likely thing to break a demo.
- **Ingest the corpus in advance, never during.** A fresh VM has an empty
  corpus, and an empty corpus does not error — the pipeline answers from the
  model's general knowledge instead, which is exactly the ungrounded answer a
  demo should not show. Confirm the app's chunk/paper counts are non-zero.
- **Warm the path you will demo.** Run one citation and one verification
  beforehand so the first live action is not also the first cold call.
- **Do not start a batch ingest during the demo.** GROBID will take two cores
  for the duration; the UI stays responsive at 4 vCPU but throughput does not.

## 9. Out of scope

- **Terraform.** Shell scripts are more readable for one VM; this is not a
  fleet.
- **CI/CD.** `deploy.sh` is run by hand.
- **Multi-user.** `ARCHITECTURE.md` §9 records the single-user assumption;
  ingestion still runs synchronously in the request.
- **Backups.** The PDFs are the source of truth and the corpus is rebuildable.
  A PD snapshot schedule is a one-line addition if wanted later.
- **Migrating the existing corpus.** Impossible across embedding models (§4.4).

## 10. Risks

**A changing source IP is the most likely demo failure.** The firewall pins
access to one address, and presenting from an unfamiliar network silently locks
you out — the app is up, healthy, and unreachable. Mitigation: `allow-ip.sh`
fixes it in one argument-free command, and §8's checklist puts running it before
the demo rather than during it. Residual risk: if the venue blocks outbound
tcp:8080, no firewall change helps; test this on arrival, not at showtime.

**Plain HTTP.** The firewall protects the OpenAI key's *effects* but the session
itself is unencrypted in transit. Acceptable for one user on one address;
unacceptable the moment this is shared. If a domain becomes available, the
upgrade path is IAP + a managed certificate on an HTTPS load balancer (~$18/mo),
and nothing else in this spec changes.

**Cost accrues whether or not the VM is used.** ≈$95/mo running. Stopping the VM
between demos zeroes the compute but the 50 GB disk keeps billing at ~$6/mo, and
a stopped VM is not a deleted VM. Set a budget alert on the project; the OpenAI
key needs its own spend limit on the OpenAI account, which this repo cannot
enforce.

**Re-ingest is a manual step that is easy to forget.** A fresh VM has an empty
corpus, and the app will answer from general knowledge rather than erroring —
the same ungrounded-answer path `ARCHITECTURE.md` already documents. Mitigation:
§8's verification list ends with an end-to-end citation run, which fails
visibly on an empty corpus.

**Model choice is now coupled to Agent 8's temperature.** `JUDGEMENT_TEMPERATURE
= 0.0` is sent unconditionally. `gpt-4.1-mini` accepts it. Switching
`CITATION_JUDGEMENT_MODEL` to a reasoning-tier model without first making that
parameter optional turns every citation into an HTTP 400, surfaced in the report
as "Unusable model reply" — a misleading label for an API rejection.
