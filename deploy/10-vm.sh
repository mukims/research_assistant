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
    --boot-disk-size=50GB --boot-disk-type=pd-balanced \
    --disk="name=${DISK_NAME},device-name=appdata,mode=rw,auto-delete=no" \
    --service-account "$SA_EMAIL" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags "$NETWORK_TAG" \
    --metadata-from-file "startup-script=${HERE}/startup.sh" \
    --project "$PROJECT"
fi

echo "==> Firewall"
MY_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || curl -fsS https://ifconfig.me 2>/dev/null || curl -fsS https://icanhazip.com 2>/dev/null || true)"
if [[ -z "$MY_IP" ]]; then
  echo "WARNING: Failed to auto-detect public IP. Run deploy/allow-ip.sh manually once connected." >&2
else
  if gcloud compute firewall-rules describe "$FIREWALL_RULE" --project "$PROJECT" >/dev/null 2>&1; then
    gcloud compute firewall-rules update "$FIREWALL_RULE" \
      --source-ranges "${MY_IP}/32" --project "$PROJECT"
  else
    gcloud compute firewall-rules create "$FIREWALL_RULE" \
      --allow tcp:8080 --source-ranges "${MY_IP}/32" \
      --target-tags "$NETWORK_TAG" --project "$PROJECT"
  fi
  echo "    tcp:8080 allowed from ${MY_IP}/32 only"
fi

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)"
echo
echo "==> VM ready. App will be at: http://${EXT_IP:-<IP-pending>}:8080"
echo "    Next: deploy/deploy.sh"
