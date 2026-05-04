# RunPod Serverless Inference Worker — Foundation Model Edition
#
# Production-hardened: multi-stage, non-root, pinned versions, minimal attack surface.
# Uses NVIDIA NIM VLM + OSM Building Footprints. No local GPU model.
# Cold-start: ~3s | Image size: ~400 MB | GPU: optional (SAM 2 fallback)
#
# IMPORTANT: build context must be the REPO ROOT (not this directory),
# so that services/api/app/ is reachable inside the context.
#
# Build (from repo root):
#   docker build --platform linux/amd64 \
#     -f infra/runpod/serverless/inference/Dockerfile \
#     -t docker.io/YOUR_USER/roof-corrosion-inference:latest \
#     .
#
# Deploy:
#   ./scripts/deploy-production.sh inference

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build

# Install uv (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /usr/local/bin/uv

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps into a venv for clean layer separation
COPY infra/runpod/serverless/inference/requirements.txt ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="roof-corrosion-inference"
LABEL org.opencontainers.image.description="Satellite roof corrosion inference — FM pipeline (NIM VLM + OSM)"
LABEL org.opencontainers.image.source="https://github.com/khiwniti/roof_corrosion_mlops"

# Install uv in runtime for any dynamic installs + runtime system deps
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy the FastAPI app package so handler.py can import from app.*
COPY --chown=appuser:appuser services/api/app/ ./src/app/

# Copy handler and test fixture
COPY --chown=appuser:appuser infra/runpod/serverless/inference/handler.py ./
COPY --chown=appuser:appuser infra/runpod/serverless/inference/test_input.json ./

# Ensure app package is importable
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime env defaults — override via RunPod endpoint secrets/env vars
ENV REGION=TH
ENV PIPELINE=fm
ENV MIN_CONFIDENCE=0.6
ENV LOG_LEVEL=INFO

# Switch to non-root user
USER appuser


CMD ["python3", "-u", "handler.py"]
