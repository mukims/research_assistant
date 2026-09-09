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
  sudo docker compose --env-file /etc/app.env ps
"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo
echo "==> Deployed. http://${EXT_IP}:8080"
echo "    If unreachable, your IP may have changed: run deploy/allow-ip.sh"
