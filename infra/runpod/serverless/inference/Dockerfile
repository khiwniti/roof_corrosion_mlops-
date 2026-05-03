# RunPod Serverless Inference Worker — Foundation Model Edition
#
# Uses NVIDIA NIM VLM + OSM Building Footprints. No local GPU model.
# Cold-start: ~3s  |  Image size: ~400 MB  |  GPU: optional (SAM 2 fallback)
#
# Build:
#   docker build --platform linux/amd64 -t roof-corrosion-inference .
#
# Push:
#   docker push YOUR_REGISTRY/roof-corrosion-inference:latest
#
# Deploy:
#   See scripts/deploy-inference.sh

FROM python:3.11-slim

WORKDIR /app

# System deps for image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libjpeg-dev libpng-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — installed first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app package so handler.py can import from app.*
# This gives us pipeline_fm, nim_client, footprint_client, region, quote_engine, etc.
COPY ../../../../services/api/app/ ./src/app/

# Copy handler and test fixture
COPY handler.py ./
COPY test_input.json ./

# Ensure app package is importable
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# RunPod-required env (override at deploy time)
ENV NVIDIA_API_KEY=""
ENV REGION=TH
ENV PIPELINE=fm
ENV MIN_CONFIDENCE=0.6

# Healthcheck — RunPod polls this
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import app.inference.pipeline_fm; print('ok')" || exit 1

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
