# syntax=docker/dockerfile:1
#
# FlyLab GPU/CPU runtime image: Ubuntu 24.04 + Python 3.12 + locked deps (uv.lock).
#
# Where CUDA comes from:
#   - CUDA *runtime* libraries (cudart, cuBLAS, ...) come from the torch wheel's pinned
#     cuda-toolkit dependency (CUDA 13.2.2 for torch 2.15.0 on Linux x86_64).
#   - The NVIDIA *driver* (libcuda) is injected at run time from the host by Docker's GPU
#     support (Docker Desktop WSL2 backend, or NVIDIA Container Toolkit on native Linux).
#   - Never install a Linux NVIDIA driver in this image or inside WSL2.
#
# Base images are pinned by digest (checked 2026-11-04). Re-pin only after a host validation run.

ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc

FROM ${UV_IMAGE} AS uv

FROM ${UBUNTU_IMAGE}

ARG FLYLAB_EXTRAS="--extra gpu"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    TRITON_CACHE_DIR=/opt/flylab/artifacts/.triton-cache \
    PATH=/opt/venv/bin:$PATH

# gcc, libc6-dev and python3.12-dev: Triton compiles a small launcher stub with the host C
# compiler the first time a kernel runs (the fused CUDA engine, flylab.neural.lif_fused).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3.12 python3.12-venv \
        gcc libc6-dev python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /opt/flylab

# Dependencies first (cached layer, ~9 GB with CUDA), project files second. Only the files the
# lock resolution reads are copied before it, so editing README/LICENSE does not rebuild it.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group dev ${FLYLAB_EXTRAS}

COPY README.md LICENSE NOTICE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev ${FLYLAB_EXTRAS}

# Ubuntu 24.04 ships an unprivileged `ubuntu` user (uid 1000), which matches the default
# WSL2 user, so files written to bind mounts stay owned by you.
RUN mkdir -p /opt/flylab/artifacts && chown ubuntu:ubuntu /opt/flylab/artifacts
USER ubuntu

# Commit of the deployed code (the host copy has no .git). Last, so it never busts the cache.
ARG FLYLAB_GIT_COMMIT=unknown
ENV FLYLAB_GIT_COMMIT=${FLYLAB_GIT_COMMIT}

CMD ["flylab", "version"]
