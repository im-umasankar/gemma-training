#!/bin/bash
# Run this once to set up GCP prerequisites before submitting the training job.
# Usage: bash setup.sh <PROJECT_ID> <REGION> <BUCKET_NAME>

set -e

PROJECT_ID="${1:?Usage: bash setup.sh <PROJECT_ID> <REGION> <BUCKET_NAME>}"
REGION="${2:-us-central1}"
BUCKET_NAME="${3:?Provide a GCS bucket name}"
REPO_NAME="gemma-training"

echo "==> Authenticating with GCP..."
gcloud config set project "$PROJECT_ID"

echo "==> Enabling required APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com

echo "==> Creating GCS bucket (if it doesn't exist)..."
gsutil mb -l "$REGION" "gs://$BUCKET_NAME" 2>/dev/null || echo "  Bucket already exists, skipping."

echo "==> Creating Artifact Registry Docker repository..."
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Gemma training images" 2>/dev/null || echo "  Repo already exists, skipping."

echo "==> Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Accept the Gemma license at: https://huggingface.co/google/gemma-3-1b"
echo "  2. Get your HF token at:        https://huggingface.co/settings/tokens"
echo "  3. Run the training job:"
echo ""
echo "     python submit_job.py \\"
echo "       --project $PROJECT_ID \\"
echo "       --region $REGION \\"
echo "       --bucket gs://$BUCKET_NAME/gemma-alpaca \\"
echo "       --hf_token hf_YOUR_TOKEN_HERE"
