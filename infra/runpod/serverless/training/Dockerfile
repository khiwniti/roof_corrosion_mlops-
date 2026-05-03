# RunPod Serverless Training Worker
# Trains SegFormer models for roof footprint and corrosion detection
#
# Build from REPO ROOT:
#   docker build --platform linux/amd64 \
#     -f infra/runpod/serverless/training/Dockerfile \
#     -t YOUR_REGISTRY/roof-corrosion-training:latest \
#     .

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps for rasterio/gdal + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ git \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps — paths relative to repo root build context
COPY infra/runpod/serverless/training/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Copy ML code
COPY ml/ /app/ml/

# Copy handler and test fixture
COPY infra/runpod/serverless/training/handler.py ./
COPY infra/runpod/serverless/training/test_input.json ./

# Data and model directories
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app
RUN mkdir -p /app/models /app/data

CMD ["python3", "-u", "handler.py"]
