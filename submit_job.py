"""
Submit a Gemma 3 1B SFT training job to Vertex AI.

Prerequisites (run once):
  gcloud auth login
  gcloud auth configure-docker <REGION>-docker.pkg.dev
  gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com

Usage:
  python submit_job.py \
    --project YOUR_PROJECT_ID \
    --region us-central1 \
    --bucket gs://your-bucket/gemma-alpaca \
    --hf_token hf_...
"""

import argparse
import subprocess
import sys

from google.cloud import aiplatform


def build_and_push_image(image_uri: str):
    print(f"Building Docker image: {image_uri}")
    subprocess.run(["docker", "build", "-t", image_uri, "."], check=True)
    print("Pushing image to Artifact Registry...")
    subprocess.run(["docker", "push", image_uri], check=True)
    print("Image pushed successfully.")


def submit_training_job(
    project: str,
    region: str,
    image_uri: str,
    gcs_output: str,
    hf_token: str,
    epochs: int,
    batch_size: int,
    model_name: str,
    machine_type: str,
    accelerator_type: str,
):
    aiplatform.init(project=project, location=region)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="gemma-3-1b-alpaca-sft",
        container_uri=image_uri,
        # Optional: track the job under a model resource
        # model_serving_container_image_uri="...",
    )

    print(f"Submitting Vertex AI training job in {region}...")
    model = job.run(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        replica_count=1,
        args=[
            "--model_name", model_name,
            "--output_dir", "/tmp/gemma-alpaca-sft",
            "--gcs_output", gcs_output,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ],
        environment_variables={
            "HF_TOKEN": hf_token,
        },
        # Sync=True waits for completion; set False to submit and return immediately
        sync=True,
    )

    print(f"Training job complete. Model resource: {model.resource_name if model else 'N/A'}")
    print(f"Model saved to: {gcs_output}")


def main():
    parser = argparse.ArgumentParser(description="Submit Gemma 3 1B SFT job to Vertex AI")

    # GCP config
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--bucket", required=True,
                        help="GCS output path, e.g. gs://my-bucket/gemma-alpaca")

    # HuggingFace
    parser.add_argument("--hf_token", required=True,
                        help="HuggingFace token (accept Gemma license at hf.co/google/gemma-3-1b)")

    # Training
    parser.add_argument("--model_name", default="google/gemma-3-1b-pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)

    # Infrastructure — T4 is cheapest; swap to L4/A100 for faster training
    parser.add_argument("--machine_type", default="n1-standard-8",
                        help="Vertex AI machine type")
    parser.add_argument("--accelerator_type", default="NVIDIA_TESLA_T4",
                        choices=["NVIDIA_TESLA_T4", "NVIDIA_L4", "NVIDIA_TESLA_A100"],
                        help="GPU accelerator type")

    # Artifact Registry repo (must exist)
    parser.add_argument("--repo", default="gemma-training",
                        help="Artifact Registry Docker repo name")

    # Skip build if image already pushed
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip Docker build/push (use if image already exists)")

    args = parser.parse_args()

    image_uri = (
        f"{args.region}-docker.pkg.dev/{args.project}/{args.repo}/gemma-3-1b-sft:latest"
    )

    if not args.skip_build:
        build_and_push_image(image_uri)

    submit_training_job(
        project=args.project,
        region=args.region,
        image_uri=image_uri,
        gcs_output=args.bucket,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
    )


if __name__ == "__main__":
    main()
