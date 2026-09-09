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
