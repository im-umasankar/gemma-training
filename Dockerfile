# CUDA 12.1 + cuDNN 8 — verified available on Docker Hub
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 (Ubuntu 22.04 ships 3.10 by default; kauldron requires >=3.11)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "jax[cuda12_pip]" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir -r requirements.txt

COPY config.py .

# Patch kauldron: gemma v3.2.1 references kd.ckpts.AbstractPartialLoader,
# which kauldron 1.4.x exposes as InitTransform — add the alias.
RUN python -c "\
import site, os; \
path = os.path.join(site.getsitepackages()[0], 'kauldron/checkpoints/__init__.py'); \
open(path, 'a').write('\nAbstractPartialLoader = InitTransform\n')"

# Entry point: Kauldron training launcher
# Data paths and workdir are passed as --cfg.* overrides by submit_job.py
ENTRYPOINT ["python", "-m", "kauldron.main", "--cfg=config.py"]
