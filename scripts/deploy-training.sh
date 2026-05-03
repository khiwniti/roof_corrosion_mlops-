#!/usr/bin/env bash
# Deploy roof corrosion training endpoint to RunPod Serverless
#
# Prerequisites:
#   - Docker installed and logged in
#   - RUNPOD_API_KEY set in .env
#
# Usage:
#   ./scripts/deploy-training.sh [REGISTRY] [TAG]

set -euo pipefail

REGISTRY="${1:-docker.io/roofcorrosion}"
TAG="${2:-latest}"
IMAGE_NAME="roof-corrosion-training"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "═══════════════════════════════════════════════════════════"
echo "  Deploying RunPod Serverless Training Endpoint"
echo "  Image: ${FULL_IMAGE}"
echo "═══════════════════════════════════════════════════════════"

# Load API key
if [ -f .env ]; then
    source .env
elif [ -f .env.local ]; then
    source .env.local
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set. Add it to .env or .env.local"
    exit 1
fi

# Build Docker image (includes ML code)
echo ""
echo "Building Docker image..."
docker build --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    -f infra/runpod/serverless/training/Dockerfile \
    --context .

# Push to registry
echo ""
echo "Pushing to registry..."
docker push "${FULL_IMAGE}"

# Create endpoint
echo ""
echo "Creating RunPod Serverless endpoint..."

RESPONSE=$(curl -s -X POST "https://rest.runpod.io/v1/endpoints" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"roof-corrosion-training\",
        \"image\": \"${FULL_IMAGE}\",
        \"gpuTypeId\": \"NVIDIA_A100_80GB\",
        \"gpuCount\": 1,
        \"vcpuCount\": 8,
        \"memoryInGb\": 32,
        \"containerDiskInGb\": 50,
        \"env\": [
            {\"key\": \"MODELS_DIR\", \"value\": \"/app/models\"},
            {\"key\": \"DATA_DIR\", \"value\": \"/app/data\"}
        ],
        \"minActiveWorkers\": 0,
        \"maxActiveWorkers\": 1,
        \"idleTimeout\": 300
    }")

ENDPOINT_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', 'ERROR'))" 2>/dev/null || echo "ERROR")

if [ "${ENDPOINT_ID}" = "ERROR" ]; then
    echo "Failed to create endpoint. Response:"
    echo "${RESPONSE}"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training Endpoint Deployed!"
echo "═══════════════════════════════════════════════════════════"
echo "  Endpoint ID: ${ENDPOINT_ID}"
echo "  GPU:         NVIDIA A100 80GB"
echo ""
echo "  Add to .env.local:"
echo "    RUNPOD_TRAINING_ENDPOINT_ID=${ENDPOINT_ID}"
echo ""
echo "  Trigger training with:"
echo "    ./scripts/run-training.sh 1   # Stage 1: roof"
echo "    ./scripts/run-training.sh 2   # Stage 2: corrosion"
echo "═══════════════════════════════════════════════════════════"
