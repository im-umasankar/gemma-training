"""
Submit a Gemma 3 1B Kauldron SFT training job to Vertex AI.

Usage:
  python submit_job.py \
    --project gemma-training-winfiny \
    --region us-central1 \
    --bucket gs://gemma-training-winfiny-data \
    --train_data gs://gemma-training-winfiny-data/data/train.json \
    --val_data gs://gemma-training-winfiny-data/data/val.json
"""

import argparse
import subprocess

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
    bucket: str,
    train_data: str,
    val_data: str,
    machine_type: str,
    accelerator_type: str,
):
    workdir = f"{bucket}/workdir"

    aiplatform.init(project=project, location=region, staging_bucket=bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="gemma-3-1b-kauldron-sft",
        container_uri=image_uri,
    )

    print(f"Submitting Vertex AI training job in {region}...")
    job.run(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        replica_count=1,
        # Kauldron CLI overrides: set workdir and data paths at launch time
        args=[
            f"--cfg.workdir={workdir}",
            f"--cfg.train_ds.data_source.path={train_data}",
            f"--cfg.evals.test.ds.data_source.path={val_data}",
            f"--cfg.evals.sampling.ds.data_source.path={val_data}",
        ],
        sync=True,
    )

    print(f"Training complete. Checkpoints saved to: {workdir}")


def main():
    parser = argparse.ArgumentParser(description="Submit Gemma 3 1B Kauldron SFT job to Vertex AI")

    # GCP
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--bucket", required=True,
                        help="GCS bucket root, e.g. gs://my-bucket")

    # Data — JSON files must be uploaded to GCS before submitting
    parser.add_argument("--train_data", required=True,
                        help="GCS path to training JSON, e.g. gs://my-bucket/data/train.json")
    parser.add_argument("--val_data", required=True,
                        help="GCS path to validation JSON, e.g. gs://my-bucket/data/val.json")

    # Infrastructure
    parser.add_argument("--machine_type", default="n1-standard-8")
    parser.add_argument("--accelerator_type", default="NVIDIA_TESLA_T4",
                        choices=["NVIDIA_TESLA_T4", "NVIDIA_L4", "NVIDIA_TESLA_A100"])

    # Artifact Registry repo
    parser.add_argument("--repo", default="gemma-training")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip Docker build/push if image already exists")

    args = parser.parse_args()

    image_uri = (
        f"{args.region}-docker.pkg.dev/{args.project}/{args.repo}/gemma-3-1b-kauldron:latest"
    )

    if not args.skip_build:
        build_and_push_image(image_uri)

    submit_training_job(
        project=args.project,
        region=args.region,
        image_uri=image_uri,
        bucket=args.bucket,
        train_data=args.train_data,
        val_data=args.val_data,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
    )


if __name__ == "__main__":
    main()
