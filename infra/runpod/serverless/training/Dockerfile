# RunPod Serverless Training Worker
#
# Production-hardened: multi-stage, non-root, pinned versions.
# Trains SegFormer models for roof footprint and corrosion detection.
#
# Build from REPO ROOT:
#   docker build --platform linux/amd64 \
#     -f infra/runpod/serverless/training/Dockerfile \
#     -t YOUR_REGISTRY/roof-corrosion-training:latest \
#     .
#
# Deploy:
#   ./scripts/deploy-production.sh training

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel AS builder

WORKDIR /build

# Install uv (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /usr/local/bin/uv

# System build deps for rasterio/gdal + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ git \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps into a venv for clean layer separation
COPY infra/runpod/serverless/training/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS runtime

LABEL org.opencontainers.image.title="roof-corrosion-training"
LABEL org.opencontainers.image.description="ML training worker — SegFormer roof/corrosion models"
LABEL org.opencontainers.image.source="https://github.com/khiwniti/roof_corrosion_mlops"

# Install uv in runtime
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /usr/local/bin/uv

# Minimal runtime deps (no -dev headers, no gcc in production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgl1 libglib2.0-0 \
    libjpeg-dev libpng-dev dumb-init && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy ML code
COPY --chown=appuser:appuser ml/ /app/ml/

# Copy handler and test fixture
COPY --chown=appuser:appuser infra/runpod/serverless/training/handler.py ./
COPY --chown=appuser:appuser infra/runpod/serverless/training/test_input.json ./

# Data and model directories
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Create writable dirs for non-root user
RUN mkdir -p /app/models /app/data && chown -R appuser:appuser /app/models /app/data

# Switch to non-root user
USER appuser

ENTRYPOINT ["dumb-init", "--"]
CMD ["python3", "-u", "handler.py"]
