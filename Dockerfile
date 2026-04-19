# CUDA 12.3 + cuDNN 9 — required by JAX cuda12 wheels
FROM nvidia/cuda:12.3.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "jax[cuda12_pip]" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir -r requirements.txt

COPY config.py .

# Entry point: Kauldron training launcher
# Data paths and workdir are passed as --cfg.* overrides by submit_job.py
ENTRYPOINT ["python", "-m", "kauldron.main", "--cfg=config.py"]
