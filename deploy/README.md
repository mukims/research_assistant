# Deploying to Google Cloud

One VM, two containers under Docker Compose, one firewall rule. See
`notimportant/superpowers/specs/2026-09-09-gcp-deployment-design.md` for architectural design rationale.

## Architecture

* **VM**: `e2-custom-4-12288` (4 vCPU, 12 GB RAM) on Debian 12
* **Storage**: 50 GB `pd-balanced` persistent disk mounted at `/mnt/disks/data`
* **Containers**:
  * `app`: Streamlit research assistant UI and Agent 0–8 pipeline (port 8080)
  * `grobid`: `grobid/grobid:0.8.1` with `JAVA_OPTS="-Xmx3g"` (internal port 8070)
* **Backend**: OpenAI for chat (`gpt-4.1-mini`) and embeddings (`text-embedding-3-small`)
* **Secrets**: `OPENAI_API_KEY` stored in Google Secret Manager, pulled into `/etc/app.env` at VM boot
* **Access**: GCE firewall rule permitting port 8080 from caller's single `/32` IP

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
