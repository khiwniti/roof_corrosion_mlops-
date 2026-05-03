# RunPod Serverless Training Worker
# Trains SegFormer models for roof footprint and corrosion detection
#
# Build:
#   docker build --platform linux/amd64 -t roof-corrosion-training .
#
# Push:
#   docker push YOUR_REGISTRY/roof-corrosion-training:latest

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

# System deps for rasterio/gdal (for real satellite data) + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ git \
    libgl1-mesa-glx libglib2.0-0 libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code
COPY ml/ /app/ml/

# Copy handler
COPY handler.py ./

# Data and model directories
ENV MODELS_DIR=/app/models
ENV DATA_DIR=/app/data
RUN mkdir -p /app/models /app/data

# AWS CLI for S3 uploads
RUN pip install --no-cache-dir awscli

CMD ["python3", "-u", "handler.py"]
