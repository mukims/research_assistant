#!/usr/bin/env bash
# VM control helper: start, stop, or check status of the research assistant VM.
# Stopping the VM drops compute billing from ~$89/mo to $0 (only the 50GB disk
# keeps billing at ~$6/mo).
source "$(dirname "$0")/config.sh"

ACTION="${1:-status}"

case "$ACTION" in
  start)
    echo "Starting VM $VM_NAME in $ZONE..."
    gcloud compute instances start "$VM_NAME" --zone "$ZONE" --project "$PROJECT"
    echo "Updating firewall to current IP..."
    "$(dirname "$0")/allow-ip.sh"
    ;;
  stop)
    echo "Stopping VM $VM_NAME in $ZONE to save compute costs..."
    gcloud compute instances stop "$VM_NAME" --zone "$ZONE" --project "$PROJECT"
    echo "VM stopped. Compute billing paused."
    ;;
  status)
    gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" \
      --format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)"
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
