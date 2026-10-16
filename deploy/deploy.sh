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
# The compose stack and the hand-started standalone container both want port
# 8080. This script used to `docker rm -f` the standalone one to make room,
# which took the live site down; it now refuses instead. Compose's own
# container is named research-assistant-app-1, so the exact-name match below
# cannot catch it.
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "
  set -e
  echo '$IMAGE' | sudo tee /etc/app-image >/dev/null
  sudo sed -i 's|^APP_IMAGE=.*|APP_IMAGE=$IMAGE|' /etc/app.env
  sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
  cd /opt/research-assistant
  if sudo docker ps -a --format '{{.Names}}' | grep -qx 'research-assistant'; then
    echo 'ERROR: a container named research-assistant already exists on this VM.' >&2
    echo 'It is the hand-started standalone image that serves the live site, and its code and corpus are not part of this compose stack.' >&2
    echo 'Not touching it. Update it with deploy/patch-standalone.sh, or stop it yourself first if you mean to replace it.' >&2
    exit 3
  fi
  sudo docker compose --env-file /etc/app.env pull
  sudo docker compose --env-file /etc/app.env up -d
  sudo docker compose --env-file /etc/app.env ps
"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo
echo "==> Deployed. http://${EXT_IP}:8080"
echo "    If unreachable, your IP may have changed: run deploy/allow-ip.sh"
