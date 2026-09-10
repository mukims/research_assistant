# GCP Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the research assistant — including Agent 8 — to a firewalled GCE VM running the app and GROBID under Docker Compose, with the corpus on a persistent disk and OpenAI for chat and embeddings.

**Architecture:** One `e2-custom-4-12288` VM. A startup script mounts a 50 GB persistent disk at `CITATION_DATA_DIR`, pulls the OpenAI key from Secret Manager, and brings up two containers via Compose: the app (built from this repo) and `grobid/grobid:0.8.1`. Access is a single firewall rule pinned to one source IP over plain HTTP. No load balancer, no domain, no Ollama at runtime.

**Tech Stack:** Docker + Docker Compose, Google Compute Engine, Artifact Registry, Secret Manager, `gcloud` CLI, bash.

**Spec:** `notimportant/superpowers/specs/2026-09-09-gcp-deployment-design.md`

## Global Constraints

- **Machine type is `e2-custom-4-12288`** — 4 vCPU, 12 GB. Not a standard SKU; the literal string matters.
- **Disk:** 50 GB `pd-balanced`, mounted at `/mnt/disks/data`, bind-mounted into the app container at `/home/user/data` (= `CITATION_DATA_DIR`).
- **Backend env, exactly:** `LLM_BACKEND=openai`, `CITATION_EMBED_BACKEND=openai`, `OPENAI_BASE_URL=https://api.openai.com/v1`, `CITATION_LLM_MODEL=gpt-4.1-mini`, `CITATION_EMBED_MODEL=text-embedding-3-small`.
- **`GROBID_SERVER=http://grobid:8070`** — the Compose service name, not localhost.
- **GROBID JVM capped:** `JAVA_OPTS=-Xmx3g`. Explicit, not default.
- **The app container never gets the Docker socket.** GROBID's Start/Stop buttons are expected to fail; only the status probe works.
- **`OPENAI_API_KEY` is never written into the image, the compose file, or any file committed to git.** It is read from Secret Manager at VM boot.
- **Secrets never appear in `argv`** — prompt for them, or pipe via stdin. A secret in an argument lands in shell history and every process listing.
- **Firewall never uses `0.0.0.0/0`.** One `/32` source range.
- **App container runs as uid 1000** (the image's `user`); the mounted disk must be `chown 1000:1000`.
- **No application code changes.** This plan touches packaging, container, deploy and docs files only. `research_assistant/` and `app.py` are not modified.
- Python 3.10–3.12 is the project's declared range (`requires-python = ">=3.10,<3.13"`).
- `shellcheck` is **not installed** on this machine — verify scripts with `bash -n` instead.
- The active gcloud project is `techireland` and **the Compute Engine API is not yet enabled on it**.

## Phase boundary — read before starting

**Tasks 1–5 create and verify repo artifacts. They cost nothing and touch no cloud resources.**

**Task 6 provisions real GCP infrastructure and starts billing (~$95/mo).** It must not run without the user's explicit go-ahead at that point. If you are an agent executing this plan, stop at the end of Task 5 and report; Task 6 is the user's decision, not yours.

---

### Task 1: Packaging correctness

**Files:**
- Modify: `pyproject.toml`
- Modify: `.dockerignore`
- Test: `tests/test_packaging.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package that carries `research_assistant/judgement/prompt.md` and `research_assistant/judgement/cases/cases.jsonl`. Task 2's image build depends on this being right.

Why this is first: `judge.py` reads `prompt.md` at **module** scope, and `app.py` imports the verifier at module scope. Under a non-editable install with the data files missing, the Streamlit app fails to *start* — not merely Agent 8. The image uses `pip install -e .` today, which masks it. `cases/` has no `__init__.py`, so `[tool.setuptools.packages.find]` will never pick it up; `package-data` is the correct mechanism.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
"""Packaging guards.

research_assistant/judgement ships two non-Python files that judge.py reads at
MODULE scope. app.py imports the verifier at module scope too, so if these are
missing from an install the Streamlit app fails to start — not just Agent 8.
setuptools does not ship non-Python files unless told to, and cases/ has no
__init__.py so packages.find never sees it. These tests guard the declaration.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


class TestJudgementDataFilesAreDeclared(unittest.TestCase):
    def setUp(self):
        self.text = PYPROJECT.read_text(encoding="utf-8")

    def test_package_data_section_exists(self):
        self.assertIn(
            "[tool.setuptools.package-data]",
            self.text,
            "pyproject.toml must declare package-data or the judgement "
            "prompt and cases are omitted from any non-editable install.",
        )

    def test_prompt_is_declared(self):
        section = self.text.split("[tool.setuptools.package-data]", 1)[-1]
        self.assertIn("prompt.md", section)

    def test_cases_are_declared(self):
        section = self.text.split("[tool.setuptools.package-data]", 1)[-1]
        self.assertTrue(
            re.search(r"cases/\*\.jsonl|cases\.jsonl", section),
            "cases/cases.jsonl must be covered by a package-data pattern.",
        )


class TestDataFilesExistWhereJudgeExpectsThem(unittest.TestCase):
    """The paths judge.py resolves at import time."""

    def test_prompt_present(self):
        self.assertTrue(
            (REPO_ROOT / "research_assistant" / "judgement" / "prompt.md").is_file()
        )

    def test_cases_present(self):
        self.assertTrue(
            (REPO_ROOT / "research_assistant" / "judgement" / "cases" / "cases.jsonl").is_file()
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python3 -m pytest tests/test_packaging.py -v`
Expected: the three `TestJudgementDataFilesAreDeclared` tests FAIL (no `package-data` section yet); the two `TestDataFilesExistWhereJudgeExpectsThem` tests PASS.

- [ ] **Step 3: Add the package-data declaration**

In `pyproject.toml`, immediately after the existing `[tool.setuptools.packages.find]` block, add:

```toml
# judge.py reads prompt.md at module scope, and app.py imports the verifier at
# module scope — so a wheel without these files fails at app startup, not at
# first use. cases/ has no __init__.py, so packages.find never sees it; this is
# the only mechanism that ships them.
[tool.setuptools.package-data]
"research_assistant.judgement" = ["prompt.md", "cases/*.jsonl"]
```

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 python3 -m pytest tests/test_packaging.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Prove the declaration actually works, by building a wheel**

The tests guard the declaration; this proves the declaration ships the files.

```bash
cd /tmp && rm -rf wheelcheck && mkdir wheelcheck && cd wheelcheck
python3 -m pip wheel --no-deps --no-build-isolation \
  -w . /run/media/shardul/storage/research_assistant
python3 -c "
import zipfile, glob
w = glob.glob('*.whl')[0]
names = zipfile.ZipFile(w).namelist()
want = ['research_assistant/judgement/prompt.md',
        'research_assistant/judgement/cases/cases.jsonl']
for f in want:
    print(('OK  ' if f in names else 'MISS'), f)
assert all(f in names for f in want), 'package-data did not ship the files'
print('wheel:', w)
"
```

Expected: both lines print `OK`, and the assert does not fire. Paste this output into your report — it is the evidence for this task.

If `pip wheel` fails for an environment reason (no `setuptools`, sandbox restriction), say so in your report and note the tests still pass; do not skip silently.

- [ ] **Step 6: Fix the stale `.dockerignore` paths**

`.dockerignore` lists top-level paths that no longer exist — everything moved under `data/` when `DATA_DIR` was introduced. Replace the block from `# Local model + runtime data` to the end of that group with:

```
# Local model + runtime data — never bake these into the image
model_final.pth
data/

# Superpowers scratch and worktrees
.superpowers/
.worktrees/

# Layout extras are not installed in the deployed image
requirements-layout.txt
```

The `data/` entry already covers `physics_vectordb/`, `bm25_index.pkl`, `pulled_pdfs/`, `drafts/`, `images/`, `logs/`, `raw/` and every manifest, because all of them live under `data/` now. The removed lines were dead.

- [ ] **Step 7: Verify the build context shrank**

```bash
cd /run/media/shardul/storage/research_assistant
du -sh --exclude=.git --exclude=data --exclude=model_final.pth --exclude=.superpowers --exclude=.worktrees .
```

Expected: single-digit MB. Record the number in your report.

- [ ] **Step 8: Run the full suite and commit**

```bash
CITATION_LOG_FILE=0 python3 -m pytest tests/ -q
git add pyproject.toml .dockerignore tests/test_packaging.py
git commit -m "build: ship judgement data files and prune stale dockerignore paths

judge.py reads prompt.md at module scope and app.py imports the verifier at
module scope, so a non-editable install without package-data fails at app
startup rather than at first use.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: 372 passed (367 + 5 new), 1 skipped.

---

### Task 2: Dockerfile — OpenAI backend and port 8080

**Files:**
- Modify: `Dockerfile` (the `ENV` block, `EXPOSE`, and the comment above them)

**Interfaces:**
- Consumes: Task 1's `package-data` declaration.
- Produces: an image whose baked-in defaults are the OpenAI backend and `PORT=8080`. Task 3's `docker-compose.yml` assumes port 8080 and overrides only the secret.

- [ ] **Step 1: Replace the ENV block and its comment**

The current comment describes a Cloud Run / Hugging Face deployment that no longer applies. Replace from `# Baked-in backend config.` through `EXPOSE 7860` with:

```dockerfile
# Baked-in backend config. Corpus + downloads land in CITATION_DATA_DIR, which
# sits outside the app's own directory tree so a read-only image still runs; on
# the GCE deployment a persistent disk is bind-mounted there. Chat and
# embeddings both go to OpenAI — no Ollama package is imported at runtime under
# these settings, since every ollama import in shared/llm.py is function-local
# and behind a backend branch. OPENAI_API_KEY is NOT baked in; it arrives from
# Secret Manager at container start.
ENV CITATION_DATA_DIR=/home/user/data \
    CITATION_LAYOUT_DETECTION=0 \
    CITATION_LOG_FILE=0 \
    LLM_BACKEND=openai \
    CITATION_EMBED_BACKEND=openai \
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    CITATION_LLM_MODEL=gpt-4.1-mini \
    CITATION_EMBED_MODEL=text-embedding-3-small \
    PORT=8080
RUN mkdir -p /home/user/data

EXPOSE 8080
```

Leave the `HEALTHCHECK` and `CMD` lines exactly as they are — both already interpolate `${PORT}`.

- [ ] **Step 2: Build the image**

```bash
cd /run/media/shardul/storage/research_assistant
docker build -t research-assistant:local .
```

Expected: a successful build. This is the slowest step in the plan (installing torch-free requirements still takes minutes); let it finish.

- [ ] **Step 3: Verify the baked-in env is exactly right**

```bash
docker inspect research-assistant:local \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | sort | grep -E 'CITATION|LLM_BACKEND|OPENAI|PORT'
```

Expected to include, verbatim:

```
CITATION_DATA_DIR=/home/user/data
CITATION_EMBED_BACKEND=openai
CITATION_EMBED_MODEL=text-embedding-3-small
CITATION_LAYOUT_DETECTION=0
CITATION_LLM_MODEL=gpt-4.1-mini
CITATION_LOG_FILE=0
OPENAI_BASE_URL=https://api.openai.com/v1
PORT=8080
```

And `LLM_BACKEND=openai`. Confirm no `OPENAI_API_KEY` and no `HF_TOKEN` appear.

- [ ] **Step 4: Verify no Ollama import on the image's own settings**

```bash
docker run --rm research-assistant:local python3 -c "
import sys
import app, orchestrate, watch
import research_assistant.agents.agent8_verifier
print('ollama:', 'ollama' in sys.modules, '| langchain_ollama:', 'langchain_ollama' in sys.modules)
" 2>&1 | tail -2
```

Expected: `ollama: False | langchain_ollama: False`. Streamlit bare-mode warnings on stderr are expected.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "build: point the image at OpenAI and port 8080

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `docker-compose.yml` and `.env.example`

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

**Interfaces:**
- Consumes: Task 2's image (`PORT=8080`, OpenAI env baked in).
- Produces: a two-service stack. Task 4's `startup.sh` runs `docker compose up -d` against this file and supplies `/etc/app.env`.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
# Two containers on one VM. The app image carries its own backend config (see
# Dockerfile); only the secret and the GROBID address are supplied here.
#
# GROBID is a separate service rather than a second process in the app image:
# it is a ~700 MB Java server that takes 30-60s to start, and running two
# servers in one container is awkward to supervise. Compose's restart policy
# replaces the app's own Start/Stop buttons, which cannot work without the
# Docker socket — deliberately not mounted (it would grant the app
# root-equivalent control of the VM).

services:
  app:
    image: ${APP_IMAGE:-research-assistant:local}
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      # Supplied at boot from Secret Manager via /etc/app.env.
      OPENAI_API_KEY: ${OPENAI_API_KEY:?OPENAI_API_KEY must be set}
      # Compose service name, not localhost: the app talks to grobid over the
      # compose network.
      GROBID_SERVER: http://grobid:8070
      UNPAYWALL_EMAIL: ${UNPAYWALL_EMAIL:-}
    volumes:
      # The persistent disk. CITATION_DATA_DIR is /home/user/data in the image.
      - /mnt/disks/data:/home/user/data
    depends_on:
      - grobid

  grobid:
    image: grobid/grobid:0.8.1
    restart: unless-stopped
    environment:
      # Explicit, not default: 12 GB total on the VM, and the app needs ~3 GB.
      JAVA_OPTS: "-Xmx3g"
    expose:
      - "8070"
```

Note `grobid` uses `expose`, not `ports` — it is reachable from the app over the compose network but never published to the host, so the firewall only ever has one port to protect.

- [ ] **Step 2: Create `.env.example`**

```bash
# Copy to .env for local compose runs. On the VM this file is generated at boot
# by deploy/startup.sh, which reads the key from Secret Manager.
#
# NEVER commit a filled-in .env. .gitignore covers .env; verify before staging.

# Required. The OpenAI API key the app uses for chat and embeddings.
OPENAI_API_KEY=sk-REPLACE_ME

# Optional. Which image compose should run. Defaults to the locally built tag;
# deploy.sh sets this to the Artifact Registry path.
APP_IMAGE=research-assistant:local

# Optional but polite. Unpaywall requires a contact address on every request,
# and OpenAlex/Crossref use it to route you to their faster pools. Leaving it
# unset falls back to config.py's placeholder, which is somebody else's address.
UNPAYWALL_EMAIL=you@example.com
```

- [ ] **Step 3: Ensure `.env` cannot be committed**

```bash
cd /run/media/shardul/storage/research_assistant
grep -qxF '.env' .gitignore || printf '\n# Local secrets — never commit\n.env\n' >> .gitignore
git check-ignore -q .env && echo ".env is ignored" || echo "WARNING: .env NOT ignored"
```

Expected: `.env is ignored`.

- [ ] **Step 4: Validate the compose file**

```bash
cd /run/media/shardul/storage/research_assistant
OPENAI_API_KEY=dummy docker compose config >/dev/null && echo "compose config valid"
OPENAI_API_KEY=dummy docker compose config | grep -E "GROBID_SERVER|JAVA_OPTS|/mnt/disks/data|8080"
```

Expected: `compose config valid`, and the grep shows `GROBID_SERVER: http://grobid:8070`, `JAVA_OPTS: -Xmx3g`, the `/mnt/disks/data:/home/user/data` bind, and the `8080:8080` port mapping.

- [ ] **Step 5: Verify the missing-secret guard fires**

```bash
cd /run/media/shardul/storage/research_assistant
docker compose config >/dev/null 2>&1 && echo "BAD: accepted a missing key" || echo "OK: refuses without OPENAI_API_KEY"
```

Expected: `OK: refuses without OPENAI_API_KEY`. The `:?` syntax makes a missing key a startup failure rather than a container that runs and 401s on every call.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml .env.example .gitignore
git commit -m "deploy: add compose stack for app + grobid

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Deployment scripts and `.gcloudignore`

**Files:**
- Create: `.gcloudignore`
- Create: `deploy/00-prereqs.sh`, `deploy/10-vm.sh`, `deploy/allow-ip.sh`, `deploy/deploy.sh`, `deploy/startup.sh`
- Create: `deploy/README.md`

**Interfaces:**
- Consumes: Task 3's `docker-compose.yml`.
- Produces: the executable path from an empty project to a running VM. Task 6 runs these.

Every script sources a shared config block so names are defined once. All are idempotent: re-running must not error or duplicate resources.

- [ ] **Step 1: Create `.gcloudignore`**

```
.git/
.gitignore
__pycache__/
*.pyc
.venv/
venv/

# Never upload: 1.2 GB corpus and an 856 MB checkpoint
data/
model_final.pth

# Scratch
.superpowers/
.worktrees/
.pytest_cache/

# Local secrets
.env
```

Without this, every build uploads ~2 GB instead of a few MB.

- [ ] **Step 2: Create `deploy/config.sh` (sourced by the others)**

```bash
#!/usr/bin/env bash
# Shared configuration. Override any of these in the environment.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west1}"
ZONE="${ZONE:-europe-west1-b}"

VM_NAME="${VM_NAME:-research-assistant}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-custom-4-12288}"   # 4 vCPU, 12 GB
DISK_NAME="${DISK_NAME:-research-assistant-data}"
DISK_SIZE="${DISK_SIZE:-50GB}"
DISK_TYPE="${DISK_TYPE:-pd-balanced}"

SA_NAME="${SA_NAME:-research-assistant-vm}"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

AR_REPO="${AR_REPO:-research-assistant}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/app:latest"

SECRET_NAME="${SECRET_NAME:-openai-api-key}"
FIREWALL_RULE="${FIREWALL_RULE:-allow-research-assistant}"
NETWORK_TAG="${NETWORK_TAG:-research-assistant}"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: no GCP project set. Run 'gcloud config set project <id>' or export PROJECT." >&2
  exit 1
fi
```

- [ ] **Step 3: Create `deploy/00-prereqs.sh`**

```bash
#!/usr/bin/env bash
# Enable APIs, create the Artifact Registry repo, and store the OpenAI key.
# Idempotent: safe to re-run.
source "$(dirname "$0")/config.sh"

echo "Project: $PROJECT   Region: $REGION"

echo "==> Enabling APIs (this can take a couple of minutes)"
gcloud services enable \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT"

echo "==> Artifact Registry repo"
if gcloud artifacts repositories describe "$AR_REPO" \
     --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  echo "    exists, skipping"
else
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker --location "$REGION" \
    --description="Research assistant app image" --project "$PROJECT"
fi

echo "==> Secret: $SECRET_NAME"
if gcloud secrets describe "$SECRET_NAME" --project "$PROJECT" >/dev/null 2>&1; then
  echo "    exists. To rotate: gcloud secrets versions add $SECRET_NAME --data-file=-"
else
  # Prompted, never an argument: a secret in argv lands in shell history and
  # in every process listing on the machine.
  echo -n "    Paste your OpenAI API key (input hidden): "
  read -rs OPENAI_KEY
  echo
  printf '%s' "$OPENAI_KEY" | gcloud secrets create "$SECRET_NAME" \
    --data-file=- --replication-policy=automatic --project "$PROJECT"
  unset OPENAI_KEY
fi

echo "==> Done. Next: deploy/10-vm.sh"
```

- [ ] **Step 4: Create `deploy/10-vm.sh`**

```bash
#!/usr/bin/env bash
# Create the service account, persistent disk, VM and firewall rule.
# Idempotent: safe to re-run.
source "$(dirname "$0")/config.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Service account"
if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  echo "    exists, skipping"
else
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Research assistant VM" --project "$PROJECT"
fi

# Least privilege: read the one secret, pull from Artifact Registry. Nothing else.
for ROLE in roles/secretmanager.secretAccessor roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${SA_EMAIL}" --role "$ROLE" \
    --condition=None --quiet >/dev/null
done
echo "    roles bound"

echo "==> Persistent disk ($DISK_SIZE $DISK_TYPE)"
if gcloud compute disks describe "$DISK_NAME" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "    exists, skipping"
else
  gcloud compute disks create "$DISK_NAME" \
    --size "$DISK_SIZE" --type "$DISK_TYPE" --zone "$ZONE" --project "$PROJECT"
fi

echo "==> VM ($MACHINE_TYPE)"
if gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "    exists, skipping. To apply a new startup script:"
  echo "      gcloud compute instances add-metadata $VM_NAME --zone $ZONE \\"
  echo "        --metadata-from-file startup-script=$HERE/startup.sh"
else
  gcloud compute instances create "$VM_NAME" \
    --zone "$ZONE" --machine-type "$MACHINE_TYPE" \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=20GB --boot-disk-type=pd-balanced \
    --disk="name=${DISK_NAME},device-name=appdata,mode=rw,auto-delete=no" \
    --service-account "$SA_EMAIL" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags "$NETWORK_TAG" \
    --metadata-from-file "startup-script=${HERE}/startup.sh" \
    --project "$PROJECT"
fi

echo "==> Firewall"
MY_IP="$(curl -fsS https://api.ipify.org)"
if gcloud compute firewall-rules describe "$FIREWALL_RULE" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules update "$FIREWALL_RULE" \
    --source-ranges "${MY_IP}/32" --project "$PROJECT"
else
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --allow tcp:8080 --source-ranges "${MY_IP}/32" \
    --target-tags "$NETWORK_TAG" --project "$PROJECT"
fi
echo "    tcp:8080 allowed from ${MY_IP}/32 only"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo
echo "==> VM ready. App will be at: http://${EXT_IP}:8080"
echo "    Next: deploy/deploy.sh"
```

- [ ] **Step 5: Create `deploy/allow-ip.sh`**

```bash
#!/usr/bin/env bash
# Repoint the firewall at wherever you are right now. Takes no arguments.
# Run this on arrival at a demo venue, after switching networks, or any time
# the app becomes unreachable — a changed source IP is the most likely cause.
source "$(dirname "$0")/config.sh"

MY_IP="$(curl -fsS https://api.ipify.org)"
echo "Current public IP: $MY_IP"

gcloud compute firewall-rules update "$FIREWALL_RULE" \
  --source-ranges "${MY_IP}/32" --project "$PROJECT"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)"
echo "Allowed. App: http://${EXT_IP:-<vm not found>}:8080"
```

- [ ] **Step 6: Create `deploy/startup.sh`**

This runs as VM metadata on every boot. It must be safe to re-run.

```bash
#!/usr/bin/env bash
# GCE startup script. Runs as root on every boot.
set -euo pipefail
exec > >(tee -a /var/log/research-assistant-startup.log) 2>&1
echo "=== startup $(date -Is) ==="

DEVICE=/dev/disk/by-id/google-appdata
MOUNT=/mnt/disks/data

echo "--> Docker"
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

echo "--> Persistent disk"
mkdir -p "$MOUNT"
# Format ONLY if there is no filesystem. blkid is the guard: without it a
# reboot would wipe the corpus.
if ! blkid "$DEVICE" >/dev/null 2>&1; then
  echo "    no filesystem found — formatting (first boot)"
  mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEVICE"
fi
grep -q "$MOUNT" /etc/fstab || \
  echo "$DEVICE $MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
mountpoint -q "$MOUNT" || mount "$MOUNT"
# The app image runs as uid 1000; the mount must be writable by it.
chown -R 1000:1000 "$MOUNT"
echo "    mounted: $(df -h "$MOUNT" | tail -1)"

echo "--> Secret"
PROJECT="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/project/project-id)"
OPENAI_API_KEY="$(gcloud secrets versions access latest \
  --secret=openai-api-key --project "$PROJECT")"
umask 077
cat > /etc/app.env <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
APP_IMAGE=$(cat /etc/app-image 2>/dev/null || echo research-assistant:local)
UNPAYWALL_EMAIL=$(cat /etc/app-email 2>/dev/null || echo "")
EOF
unset OPENAI_API_KEY
echo "    /etc/app.env written (mode $(stat -c %a /etc/app.env))"

echo "--> Compose"
if [[ -f /opt/research-assistant/docker-compose.yml ]]; then
  cd /opt/research-assistant
  # Authenticate to Artifact Registry, deriving the host from the image path
  # rather than hardcoding a region.
  REGISTRY="$(sed -n 's|^APP_IMAGE=\([^/]*\)/.*|\1|p' /etc/app.env)"
  if [[ "$REGISTRY" == *docker.pkg.dev ]]; then
    gcloud auth configure-docker "$REGISTRY" --quiet || true
  fi
  docker compose --env-file /etc/app.env pull || true
  docker compose --env-file /etc/app.env up -d
  docker compose --env-file /etc/app.env ps
else
  echo "    /opt/research-assistant/docker-compose.yml not present yet."
  echo "    Run deploy/deploy.sh from your workstation."
fi
echo "=== startup complete $(date -Is) ==="
```

- [ ] **Step 7: Create `deploy/deploy.sh`**

```bash
#!/usr/bin/env bash
# Build, push, and roll out. Run from your workstation.
source "$(dirname "$0")/config.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Building $IMAGE"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build -t "$IMAGE" "$ROOT"

echo "==> Pushing"
docker push "$IMAGE"

echo "==> Shipping compose file"
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command \
  "sudo mkdir -p /opt/research-assistant && sudo chown \$USER /opt/research-assistant"
gcloud compute scp "$ROOT/docker-compose.yml" \
  "${VM_NAME}:/opt/research-assistant/docker-compose.yml" \
  --zone "$ZONE" --project "$PROJECT"

echo "==> Rolling out"
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "
  set -e
  echo '$IMAGE' | sudo tee /etc/app-image >/dev/null
  sudo sed -i 's|^APP_IMAGE=.*|APP_IMAGE=$IMAGE|' /etc/app.env
  sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
  cd /opt/research-assistant
  sudo docker compose --env-file /etc/app.env pull
  sudo docker compose --env-file /etc/app.env up -d
  sudo docker compose ps
"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo
echo "==> Deployed. http://${EXT_IP}:8080"
echo "    If unreachable, your IP may have changed: run deploy/allow-ip.sh"
```

- [ ] **Step 8: Create `deploy/README.md`**

```markdown
# Deploying

One VM, two containers, one firewall rule. See
`notimportant/superpowers/specs/2026-09-09-gcp-deployment-design.md` for why.

## First time

```bash
./deploy/00-prereqs.sh   # APIs, Artifact Registry, OpenAI key (prompts)
./deploy/10-vm.sh        # service account, disk, VM, firewall
./deploy/deploy.sh       # build, push, roll out
```

## Every time after

```bash
./deploy/deploy.sh
```

## When the app is unreachable

Almost always a changed source IP — the firewall pins access to one address.

```bash
./deploy/allow-ip.sh
```

## Before a demo

- Start the VM **ten minutes early**: GROBID's JVM takes 30–60 s and the first
  request after boot is the slowest.
- Run `./deploy/allow-ip.sh` **from the venue's network**.
- Confirm the corpus is non-empty. An empty corpus does not error — the app
  answers from the model's general knowledge instead, which is exactly the
  ungrounded answer a demo should not show.
- Run one citation and one verification to warm the path.
- Do not start a batch ingest during the demo.

## Cost

≈$95/mo running (VM ~$89 + 50 GB disk ~$6). Stopping the VM zeroes the compute;
the disk keeps billing.

```bash
gcloud compute instances stop research-assistant --zone europe-west1-b
gcloud compute instances start research-assistant --zone europe-west1-b
```
```

- [ ] **Step 9: Make the scripts executable and syntax-check them**

`shellcheck` is not installed on this machine, so use bash's own parser.

```bash
cd /run/media/shardul/storage/research_assistant
chmod +x deploy/*.sh
for f in deploy/*.sh; do bash -n "$f" && echo "OK  $f" || echo "FAIL $f"; done
```

Expected: `OK` for all six.

- [ ] **Step 10: Verify no secret is ever passed as an argument**

```bash
grep -n "secrets create\|secrets versions add\|OPENAI_API_KEY=" deploy/*.sh
```

Expected: the only `secrets create` uses `--data-file=-` fed from a piped `printf`, and no line passes a key literal on a command line. Confirm by reading the matches.

- [ ] **Step 11: Commit**

```bash
git add .gcloudignore deploy/
git commit -m "deploy: add GCE provisioning and rollout scripts

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `ARCHITECTURE.md` (§8 — currently describes a Cloud Run target that is wrong)
- Modify: `HOW_TO_USE.md` (the persistence table)
- Modify: `README.md` (one pointer to `deploy/`)

- [ ] **Step 1: Replace ARCHITECTURE.md §8.2**

The current §8.2 says Cloud Run fits and that the corpus persists "if a GCS volume is mounted at `CITATION_DATA_DIR`". That is the claim this deployment disproves. Replace the §8.2 body with:

```markdown
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
```

Renumber the existing §8.3 (file logging) and §8.4 (Streamlit) to §8.4 and §8.5.

- [ ] **Step 2: Update the HOW_TO_USE.md persistence table**

Replace the `| Cloud Run | In memory, **lost when the instance scales to zero** ... |` row with:

```markdown
| GCE VM (`deploy/`) | A 50 GB persistent disk mounted at `/mnt/disks/data` — survives reboots, stops and redeploys |
```

- [ ] **Step 3: Add a README pointer**

After the Docker mention in the README's intro paragraph, add:

```markdown
Deploying to Google Cloud is scripted in [deploy/](deploy/) — one VM running the
app and GROBID under Compose, with the corpus on a persistent disk. See
[deploy/README.md](deploy/README.md).
```

- [ ] **Step 4: Check the section numbering is gapless and cross-references still resolve**

```bash
cd /run/media/shardul/storage/research_assistant
grep -n "^## [0-9]" ARCHITECTURE.md
grep -n "§8\.[0-9]" ARCHITECTURE.md README.md HOW_TO_USE.md
```

Expected: `## 1.` through `## 11.` with no gaps, and every `§8.x` reference pointing at a subsection that exists.

- [ ] **Step 5: Verify no stale Cloud Run claims survive**

```bash
grep -rn "Cloud Run" ARCHITECTURE.md README.md HOW_TO_USE.md
```

Expected: mentions only in the rejected-alternatives context of §8.2. Any sentence still telling the reader to deploy on Cloud Run, or that a GCS volume makes the corpus persist, is a defect — fix it.

- [ ] **Step 6: Commit**

```bash
git add ARCHITECTURE.md HOW_TO_USE.md README.md
git commit -m "docs: replace the Cloud Run deployment section with the GCE design

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Provision and verify — REQUIRES EXPLICIT USER GO-AHEAD

**This task creates billable infrastructure (~$95/mo) and must not be started without the user saying so at this point in time. Earlier approval of the plan is not approval of this task.**

**Files:** none — this runs the scripts from Task 4.

- [ ] **Step 1: Confirm the target project with the user**

```bash
gcloud config get-value project
```

The active project is `techireland`. Confirm with the user that this is where they want the VM, and that billing is enabled on it. Do not proceed on an assumption.

- [ ] **Step 2: Prerequisites**

```bash
./deploy/00-prereqs.sh
```

Enables three APIs (Compute Engine is **not currently enabled** on `techireland`), creates the Artifact Registry repo, and prompts for the OpenAI key. Expect a few minutes for API enablement.

- [ ] **Step 3: Provision the VM**

```bash
./deploy/10-vm.sh
```

Expected: service account, 50 GB disk, `e2-custom-4-12288` VM, firewall rule pinned to your current IP. Prints the external IP.

- [ ] **Step 4: Build and roll out**

```bash
./deploy/deploy.sh
```

Expected: image pushed to Artifact Registry, compose file shipped, both containers reported by `docker compose ps`.

- [ ] **Step 5: Verify the stack is healthy on the VM**

```bash
gcloud compute ssh research-assistant --zone europe-west1-b --command "
  sudo docker compose -f /opt/research-assistant/docker-compose.yml --env-file /etc/app.env ps
  curl -fsS http://localhost:8080/_stcore/health && echo ' <- app healthy'
  curl -fsS http://localhost:8070/api/isalive && echo ' <- grobid alive'
"
```

Expected: both services `running`, and both curls succeed. GROBID may need 30–60 s after boot.

- [ ] **Step 6: Verify the firewall actually restricts access**

An untested firewall rule is an assumption, not a control. From the allowed network, `http://<EXT_IP>:8080` should load. From a phone on cellular (not wifi), it must time out. Record both results.

- [ ] **Step 7: Verify persistence survives a reboot**

```bash
gcloud compute instances stop research-assistant --zone europe-west1-b
gcloud compute instances start research-assistant --zone europe-west1-b
# wait ~90s, then:
gcloud compute ssh research-assistant --zone europe-west1-b --command "
  df -h /mnt/disks/data
  ls -la /mnt/disks/data
  sudo docker compose -f /opt/research-assistant/docker-compose.yml --env-file /etc/app.env ps
"
```

Expected: the disk is still mounted with its contents intact, and both containers came back up. **This is the step that proves `startup.sh`'s `blkid` guard works** — if the disk were reformatted on reboot, the corpus would be gone.

- [ ] **Step 8: End-to-end functional check**

In the browser: seed a paper in Tab 1, cite a draft in Tab 3, and click **Verify citations**. This one flow exercises OpenAI chat, OpenAI embeddings, ChromaDB on the persistent disk, and Agent 8 together. A verdict table means the deployment works.

- [ ] **Step 9: Ingest the corpus before any demo**

A fresh VM has an empty corpus, and an empty corpus does not error — the pipeline answers from general knowledge instead. Ingest ahead of time and confirm non-zero chunk and paper counts in the UI.

---

## Final verification

- [ ] Full suite green: `CITATION_LOG_FILE=0 python3 -m pytest tests/ -q` → 372 passed, 1 skipped
- [ ] No application code was modified:

```bash
# 239328b is main's HEAD before this plan's work began.
git diff --stat 239328b..HEAD -- research_assistant/ app.py orchestrate.py watch.py
```

Expected: empty. This plan touches packaging, container, deploy and docs only.

- [ ] No secret is committed anywhere:

```bash
git log -p --all | grep -nE "sk-[A-Za-z0-9]{20,}" && echo "SECRET FOUND — STOP" || echo "clean"
git check-ignore -q .env && echo ".env ignored"
```

- [ ] The image carries no key: `docker inspect research-assistant:local --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c OPENAI_API_KEY` → `0`
