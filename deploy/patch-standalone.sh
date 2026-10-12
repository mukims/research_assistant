#!/usr/bin/env bash
# Hot-patch code directly into the standalone container on the GCE VM.
set -euo pipefail
source "$(dirname "$0")/config.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Packaging code patch..."
tar -C "$ROOT" -czf /tmp/patch.tgz app.py orchestrate.py research_assistant tests

echo "==> Uploading patch to VM ($VM_NAME)..."
gcloud compute scp /tmp/patch.tgz "${VM_NAME}:/tmp/patch.tgz" --zone "$ZONE" --project "$PROJECT"

echo "==> Applying patch inside container and restarting Streamlit..."
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "
  set -e
  rm -rf /tmp/patch && mkdir -p /tmp/patch && tar xzf /tmp/patch.tgz -C /tmp/patch
  for f in /tmp/patch/*; do sudo docker cp \"\$f\" research-assistant:/home/user/app/; done
  sudo docker exec research-assistant chown -R 1000:1000 /home/user/app
  sudo docker restart research-assistant
  sleep 8
  sudo docker logs --tail 25 research-assistant
"

echo "==> Done. Patch deployed to standalone container."
