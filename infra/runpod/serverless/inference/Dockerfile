# RunPod Serverless Inference Worker — Foundation Model Edition
#
# Uses NVIDIA NIM VLM + OSM Building Footprints. No local GPU model.
# Cold-start: ~3s  |  Image size: ~400 MB  |  GPU: optional (SAM 2 fallback)
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
#   ./scripts/deploy-inference.sh docker.io/YOUR_USER

FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps for image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — installed first for layer caching
COPY infra/runpod/serverless/inference/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the FastAPI app package so handler.py can import from app.*
COPY services/api/app/ ./src/app/

# Copy handler and test fixture
COPY infra/runpod/serverless/inference/handler.py ./
COPY infra/runpod/serverless/inference/test_input.json ./

# Ensure app package is importable
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Runtime env defaults — override via RunPod endpoint secrets/env vars
ENV REGION=TH
ENV PIPELINE=fm
ENV MIN_CONFIDENCE=0.6

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import app.inference.pipeline_fm; print('ok')" || exit 1

CMD ["python3", "-u", "handler.py"]
