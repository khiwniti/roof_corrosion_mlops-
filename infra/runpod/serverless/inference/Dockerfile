# RunPod Serverless Inference Worker
# Two-stage roof corrosion detection: SegFormer-B3 → SegFormer-B2
#
# Build:
#   docker build --platform linux/amd64 -t roof-corrosion-inference .
#
# Push:
#   docker push YOUR_REGISTRY/roof-corrosion-inference:latest
#
# Deploy:
#   See scripts/deploy-inference.sh

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py ./

# Model directory — mount network volume here for model caching
# If models are baked into the image, COPY them into /app/models/
# Otherwise they'll be downloaded from HuggingFace on first cold start
ENV MODELS_DIR=/app/models
RUN mkdir -p /app/models

# Optional: bake TorchScript models into the image for faster cold starts
# COPY models/stage1_roof_footprint.pt /app/models/
# COPY models/stage2_corrosion.pt /app/models/

# RunPod serverless entrypoint
CMD ["python3", "-u", "handler.py"]
