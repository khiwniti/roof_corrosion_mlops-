# RunPod Serverless Production Deployment Guide

## Prerequisites

1. **Docker Hub account** with push access (images: `khiwnitigetintheq/*`)
2. **RunPod account** with API key from https://runpod.io/console/serverless
3. **NVIDIA API key** (free) from https://build.nvidia.com for NIM VLM access

## Quick Deploy

```bash
# 1. Set environment variables
export DOCKER_USERNAME=khiwnitigetintheq
export DOCKER_PASSWORD=dckr_pat_xxx  # your Docker Hub token
export RUNPOD_API_KEY=rpa_xxx        # your RunPod API key
export NVIDIA_API_KEY=nvapi_xxx      # your NVIDIA NIM key

# 2. Deploy inference endpoint (builds, pushes, creates RunPod endpoint)
./scripts/deploy-production.sh inference

# 3. Deploy training endpoint (CUDA image — may take 10-15 min)
./scripts/deploy-production.sh training

# 4. Deploy API as persistent pod
./scripts/deploy-production.sh api

# Or deploy everything at once:
./scripts/deploy-production.sh all
```

## Manual Deployment (via RunPod UI)

If you prefer the RunPod web console:

### Inference Endpoint
1. Go to https://runpod.io/console/serverless/create
2. Select "Custom Image"
3. Image: `docker.io/khiwnitigetintheq/roof-corrosion-inference:latest`
4. GPU: NVIDIA RTX A4000 (or Tesla T4 for cost savings)
5. Workers: Min 0, Max 3, Idle Timeout 300s
6. Environment Variables:
   - `REGION=TH`
   - `PIPELINE=fm`
   - `PYTHONPATH=/app/src`
   - `PYTHONUNBUFFERED=1`
   - `MIN_CONFIDENCE=0.6`
   - `LOG_LEVEL=INFO`
   - `NVIDIA_API_KEY=<your-key>`
7. Click "Create"

### Training Endpoint
1. Image: `docker.io/khiwnitigetintheq/roof-corrosion-training:latest`
2. GPU: NVIDIA A100 80GB
3. Workers: Min 0, Max 1
4. Environment Variables:
   - `MODELS_DIR=/app/models`
   - `DATA_DIR=/app/data`
   - `PYTHONUNBUFFERED=1`

## Test the Inference Endpoint

```bash
# After creating the endpoint, save the endpoint ID
export RUNPOD_ENDPOINT_ID=<from-deploy-output>

# Test with coordinates (auto-fetches satellite tile)
curl -s -X POST \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "lat": 13.7563,
      "lng": 100.5018,
      "gsd": 0.30,
      "address": "Silom Rd, Bangkok",
      "material": "corrugated_metal",
      "region": "TH"
    }
  }'
```

## Image Details

| Image | Base | Size | GPU Required | Purpose |
|-------|------|------|-------------|---------|
| roof-corrosion-inference | python:3.11.9-slim | ~400MB | Optional (SAM fallback) | FM pipeline: NIM VLM + OSM |
| roof-corrosion-training | pytorch:2.1.2-cuda12.1 | ~8GB | Yes (A100) | SegFormer training |
| roof-corrosion-api | python:3.11.9-slim | ~2GB | Optional | FastAPI + quote engine |

## CI/CD

Pushing to `main` automatically:
1. Builds all 3 images via `.github/workflows/cd.yml`
2. Pushes to Docker Hub (if secrets configured)
3. Updates RunPod endpoints (if endpoint IDs configured in GitHub vars)

Required GitHub secrets:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `RUNPOD_API_KEY`

Required GitHub variables:
- `RUNPOD_INFERENCE_ENDPOINT_ID`
- `RUNPOD_TRAINING_ENDPOINT_ID`
