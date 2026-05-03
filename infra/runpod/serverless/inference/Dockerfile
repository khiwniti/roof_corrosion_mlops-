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

# System deps for image I/O
# Note: libgl1-mesa-glx was removed in Debian 13 (trixie); use libgl1 instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — installed first for layer caching
# Paths are relative to the REPO ROOT build context
COPY infra/runpod/serverless/inference/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app package so handler.py can import from app.*
# This gives us pipeline_fm, nim_client, footprint_client, region, quote_engine, etc.
COPY services/api/app/ ./src/app/

# Copy handler and test fixture
COPY infra/runpod/serverless/inference/handler.py ./
COPY infra/runpod/serverless/inference/test_input.json ./

# Ensure app package is importable
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Runtime env defaults — override via RunPod endpoint secrets/env vars
# NVIDIA_API_KEY is NOT baked in; inject it as a RunPod secret at deploy time
ENV REGION=TH
ENV PIPELINE=fm
ENV MIN_CONFIDENCE=0.6

# Healthcheck — RunPod polls this
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import app.inference.pipeline_fm; print('ok')" || exit 1

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
