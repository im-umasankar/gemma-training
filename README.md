# Gemma 3 1B — Instruction Fine-Tuning on Google Cloud

Fine-tune Google's Gemma 3 1B model on the Stanford Alpaca dataset using Vertex AI.

## Overview

| | |
|---|---|
| **Model** | Gemma 3 1B (`google/gemma-3-1b-pt`) |
| **Method** | Supervised Fine-Tuning (SFT) + QLoRA (4-bit) |
| **Dataset** | Stanford Alpaca (52k instruction pairs) |
| **Platform** | Google Cloud Vertex AI Custom Training |
| **GPU** | NVIDIA T4 (default) |
| **Estimated cost** | ~$4–6 per full run |

---

## Project Structure

```
ai-training/
├── train.py          # Training script (runs inside the container on Vertex AI)
├── requirements.txt  # Python dependencies for the container
├── Dockerfile        # Container image definition
├── submit_job.py     # Submits the training job to Vertex AI from your machine
└── setup.sh          # One-time GCP setup script
```

---

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and authenticated
- [Docker](https://docs.docker.com/get-docker/) installed and running
- A GCP project with billing enabled
- A HuggingFace account with access to Gemma

---

## Step 1 — GCP Project Setup

The project `gemma-training-winfiny` has already been created and configured with:
- Billing account: `010C58-F2A959-19BFF5`
- GCS bucket: `gs://gemma-training-winfiny-data`
- Artifact Registry repo: `gemma-training` (us-central1)

If starting fresh with a new project, run:

```bash
# Create a new GCP project
gcloud projects create YOUR_PROJECT_ID --name="Gemma Training"

# Link billing
gcloud billing projects link YOUR_PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID

# Run the setup script (enables APIs, creates bucket and Artifact Registry repo)
bash setup.sh YOUR_PROJECT_ID us-central1 YOUR_BUCKET_NAME
```

---

## Step 2 — HuggingFace Access

Gemma is a gated model. You need to request access before downloading it.

1. Go to [huggingface.co/google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt)
2. Click **"Agree and access repository"** to accept the license
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Click **"New token"**, name it `gemma-training`, select **Read** access
5. Copy the token — you'll need it in Step 4

---

## Step 3 — Python Environment (Local)

The submission script runs locally and needs `google-cloud-aiplatform`:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependency
pip install google-cloud-aiplatform
```

---

## Step 4 — Submit the Training Job

```bash
source .venv/bin/activate

python submit_job.py \
  --project gemma-training-winfiny \
  --region us-central1 \
  --bucket gs://gemma-training-winfiny-data/gemma-alpaca \
  --hf_token hf_YOUR_TOKEN_HERE
```

This will:
1. Build the Docker image locally
2. Push it to Artifact Registry
3. Submit a Vertex AI Custom Training Job (synchronous — waits for completion)

> **Tip:** Store your token as an env var to avoid it appearing in shell history:
> ```bash
> export HF_TOKEN=hf_YOUR_TOKEN_HERE
> python submit_job.py ... --hf_token "$HF_TOKEN"
> ```

### Skip Docker build (if image already pushed)

```bash
python submit_job.py \
  --project gemma-training-winfiny \
  --region us-central1 \
  --bucket gs://gemma-training-winfiny-data/gemma-alpaca \
  --hf_token "$HF_TOKEN" \
  --skip_build
```

---

## GPU Options

| Flag | GPU | VRAM | Speed | ~Cost/hr | ~Total cost |
|---|---|---|---|---|---|
| `NVIDIA_TESLA_T4` (default) | T4 | 16 GB | ~5–7 hrs | $0.75 | **$4–6** |
| `NVIDIA_L4` | L4 | 24 GB | ~2–4 hrs | $1.40 | **$3–6** |
| `NVIDIA_TESLA_A100` | A100 | 40 GB | ~1–2 hrs | $3.80 | **$4–8** |

To use a different GPU:

```bash
python submit_job.py \
  ... \
  --accelerator_type NVIDIA_L4 \
  --machine_type g2-standard-12
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 (effective: 16 with grad accumulation) |
| Max sequence length | 512 tokens |
| Learning rate | 2e-4 (cosine schedule) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Optimizer | paged_adamw_32bit |

To change these, pass flags to `submit_job.py`:

```bash
python submit_job.py ... --epochs 5 --batch_size 8
```

---

## Output

The trained model (LoRA adapter + tokenizer) is saved to:

```
gs://gemma-training-winfiny-data/gemma-alpaca/
```

To download locally:

```bash
gsutil -m cp -r gs://gemma-training-winfiny-data/gemma-alpaca ./trained-model
```

---

## Monitor Training in GCP Console

1. Go to [console.cloud.google.com/vertex-ai/training](https://console.cloud.google.com/vertex-ai/training)
2. Select project `gemma-training-winfiny`
3. Click on the job `gemma-3-1b-alpaca-sft` to see logs and metrics

---

## Troubleshooting

**Docker build fails**
- Make sure Docker is running: `docker info`
- Re-authenticate: `gcloud auth configure-docker us-central1-docker.pkg.dev`

**HuggingFace 401 error**
- Ensure you accepted the Gemma license at huggingface.co/google/gemma-3-1b-pt
- Check your token has Read permissions

**Vertex AI quota error**
- T4 GPUs may need quota increase: go to IAM & Admin > Quotas and request `NVIDIA_TESLA_T4` in `us-central1`

**Out of memory during training**
- Reduce `--batch_size` to 2
- Or upgrade to a GPU with more VRAM (`NVIDIA_L4` or `NVIDIA_TESLA_A100`)
