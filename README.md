# Gemma 3 1B — Kauldron SFT on Google Cloud

Fine-tune Google's Gemma 3 1B model on your own JSON prompt/response dataset using [Kauldron](https://github.com/google-research/kauldron) (Google DeepMind's JAX training framework) on Vertex AI.

## Overview

| | |
|---|---|
| **Model** | Gemma 3 1B (`gm.nn.Gemma3_1B`) |
| **Checkpoint** | `gs://gemma-data/checkpoints/gemma3-1b-pt` (public) |
| **Method** | Supervised Fine-Tuning (SFT) via `gm.data.Seq2SeqTask` |
| **Framework** | [Kauldron](https://github.com/google-research/kauldron) + JAX |
| **Dataset** | Your own JSON file with `prompt`/`response` pairs |
| **Platform** | Google Cloud Vertex AI Custom Training |
| **GPU** | NVIDIA T4 (default) |
| **Estimated cost** | ~$4–6 per full run |

---

## Project Structure

```
ai-training/
├── config.py         # Kauldron training config (model, data, optimizer, evals)
├── requirements.txt  # Python dependencies for the training container
├── Dockerfile        # Container image (JAX + CUDA 12 + Kauldron + Gemma)
├── submit_job.py     # Submits the training job to Vertex AI
├── setup.sh          # One-time GCP setup (APIs, bucket, Artifact Registry)
└── data/
    └── sample.json   # Example JSON format for your dataset
```

---

## Dataset Format

Your training and validation data must be JSON files — a list of objects with `prompt` and `response` keys:

```json
[
  {
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris."
  },
  {
    "prompt": "Explain Newton's first law.",
    "response": "An object in motion stays in motion unless acted upon by an external force."
  }
]
```

See `data/sample.json` for a full example.

---

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) authenticated
- [Docker](https://docs.docker.com/get-docker/) installed and running
- GCP project `gemma-training-winfiny` with billing enabled (already set up)

---

## Step 1 — Upload Your Data to GCS

```bash
# Split your data into train and val JSON files, then upload:
gsutil cp data/train.json gs://gemma-training-winfiny-data/data/train.json
gsutil cp data/val.json   gs://gemma-training-winfiny-data/data/val.json
```

---

## Step 2 — Python Environment (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install google-cloud-aiplatform
```

---

## Step 3 — Submit the Training Job

```bash
source .venv/bin/activate

python submit_job.py \
  --project gemma-training-winfiny \
  --region us-central1 \
  --bucket gs://gemma-training-winfiny-data \
  --train_data gs://gemma-training-winfiny-data/data/train.json \
  --val_data   gs://gemma-training-winfiny-data/data/val.json
```

This will:
1. Build the Docker image and push it to Artifact Registry
2. Submit a Vertex AI Custom Training Job
3. Kauldron loads the pretrained checkpoint from `gs://gemma-data/checkpoints/gemma3-1b-pt`
4. Checkpoints are saved to `gs://gemma-training-winfiny-data/workdir/`

### Skip Docker rebuild (if image already pushed)

```bash
python submit_job.py ... --skip_build
```

---

## How It Works

The training config (`config.py`) uses:

| Component | Implementation |
|---|---|
| Data loading | `kd.data.py.Json` — reads your JSON file from GCS |
| Data transform | `gm.data.Seq2SeqTask` — tokenizes prompt/response, creates loss mask |
| Model | `gm.nn.Gemma3_1B` — JAX/Flax Gemma 3 1B |
| Pretrained weights | `gm.ckpts.LoadCheckpoint(GEMMA3_1B_PT)` — loaded from Google's public bucket |
| Loss | `SoftmaxCrossEntropyWithIntLabels` — only applied to response tokens |
| Optimizer | `optax.adafactor` |
| Checkpointing | `kd.ckpts.Checkpointer` — saves every 500 steps to GCS |

---

## Customizing the Config

Override any config value from the CLI without editing `config.py`:

```bash
# Change number of training steps
--cfg.num_train_steps=10000

# Change batch size
--cfg.train_ds.batch_size=16

# Change max sequence length
--cfg.train_ds.transforms[0].max_length=1024
```

---

## GPU Options

| Flag | GPU | VRAM | ~Cost/hr | ~Total |
|---|---|---|---|---|
| `NVIDIA_TESLA_T4` (default) | T4 | 16 GB | $0.75 | **$4–6** |
| `NVIDIA_L4` | L4 | 24 GB | $1.40 | **$3–6** |
| `NVIDIA_TESLA_A100` | A100 | 40 GB | $3.80 | **$4–8** |

```bash
python submit_job.py ... --accelerator_type NVIDIA_L4 --machine_type g2-standard-12
```

---

## Monitor Training

1. Go to [console.cloud.google.com/vertex-ai/training](https://console.cloud.google.com/vertex-ai/training)
2. Select project `gemma-training-winfiny`
3. Click `gemma-3-1b-kauldron-sft` to view logs, loss curves, and eval samples

---

## Download Checkpoints

```bash
gsutil -m cp -r gs://gemma-training-winfiny-data/workdir ./checkpoints
```

---

## Troubleshooting

**JAX CUDA error on startup**
- The container uses CUDA 12.3 + cuDNN 9; ensure your Vertex AI region supports the selected GPU.

**`gs://gemma-data` access denied**
- The Gemma checkpoints are publicly readable but require your GCP project to be set up correctly. Run `gcloud auth application-default login`.

**Out of memory**
- Reduce `batch_size` in `config.py` or switch to a higher-VRAM GPU.

**JSON key mismatch**
- Your JSON objects must have exactly `"prompt"` and `"response"` keys. If they're different (e.g. `"input"`/`"output"`), update `in_prompt` and `in_response` in `config.py`.
