#!/usr/bin/env bash
# Repoint the firewall at wherever you are right now. Takes no arguments.
# Run this on arrival at a demo venue, after switching networks, or any time
# the app becomes unreachable — a changed source IP is the most likely cause.
source "$(dirname "$0")/config.sh"

MY_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || curl -fsS https://ifconfig.me 2>/dev/null || curl -fsS https://icanhazip.com 2>/dev/null || true)"

if [[ -z "$MY_IP" ]]; then
  echo "ERROR: Could not resolve public IP address from any resolver (ipify, ifconfig.me, icanhazip)." >&2
  exit 1
fi

echo "Current public IP: $MY_IP"

gcloud compute firewall-rules update "$FIREWALL_RULE" \
  --source-ranges "${MY_IP}/32" --project "$PROJECT"

EXT_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)"
echo "Allowed. App: http://${EXT_IP:-<vm not found>}:8080"
