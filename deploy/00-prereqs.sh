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
