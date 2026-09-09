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
# Wait up to 30s for udev to create the device node to prevent false-negative blkid checks
for i in $(seq 1 30); do
  if test -b "$DEVICE"; then
    break
  fi
  sleep 1
done

if ! test -b "$DEVICE"; then
  echo "ERROR: block device $DEVICE did not appear within 30 seconds" >&2
  exit 1
fi

# Format ONLY if there is no filesystem. blkid is the guard: without it a reboot would wipe the corpus.
if ! blkid "$DEVICE" >/dev/null 2>&1; then
  echo "    no filesystem found — formatting (first boot)"
  mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEVICE"
fi
grep -q "$MOUNT" /etc/fstab || \
  echo "$DEVICE $MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
mountpoint -q "$MOUNT" || mount "$MOUNT"

# The app image runs as uid 1000; ensure mount is owned by 1000:1000.
# Only chown recursively if the root ownership does not match, avoiding boot-time IO stalls on large corpora.
if [[ "$(stat -c '%u:%g' "$MOUNT")" != "1000:1000" ]]; then
  echo "    setting ownership to 1000:1000"
  chown -R 1000:1000 "$MOUNT"
fi
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
