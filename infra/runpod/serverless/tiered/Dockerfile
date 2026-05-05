# RunPod Serverless — Tiered Hybrid Handler
# Phase 1a/3: S2/VHR ingestion → cloud mask → composite → classifier → mask
#
# Build:
#   docker build --platform linux/amd64 -t roof-corrosion-tiered:latest .
#
# Deploy to RunPod:
#   See infra/runpod/serverless/DEPLOY_GUIDE.md

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY ml/ingestion /ml/ingestion
COPY ml/inference /ml/inference
COPY ml/train/models /ml/train/models

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/ml:$PYTHONPATH

# RunPod serverless entry point
CMD ["python", "-u", "handler.py"]
