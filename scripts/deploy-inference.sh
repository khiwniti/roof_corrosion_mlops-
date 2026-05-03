#!/usr/bin/env bash
# Deploy roof corrosion inference endpoint to RunPod Serverless
#
# Prerequisites:
#   - Docker installed and logged in (docker login)
#   - RUNPOD_API_KEY set in .env
#   - Docker image built and pushed to registry
#
# Usage:
#   ./scripts/deploy-inference.sh [REGISTRY] [TAG]
#
# Examples:
#   ./scripts/deploy-inference.sh docker.io/myuser v1.0
#   ./scripts/deploy-inference.sh  # defaults: docker.io/roofcorrosion, latest

set -euo pipefail

REGISTRY="${1:-docker.io/roofcorrosion}"
TAG="${2:-latest}"
IMAGE_NAME="roof-corrosion-inference"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "═══════════════════════════════════════════════════════════"
echo "  Deploying RunPod Serverless Inference Endpoint"
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

# Build Docker image
echo ""
echo "Building Docker image..."
docker build --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    infra/runpod/serverless/inference/

# Push to registry
echo ""
echo "Pushing to registry..."
docker push "${FULL_IMAGE}"

# Create or update RunPod endpoint
echo ""
echo "Creating RunPod Serverless endpoint..."

RESPONSE=$(curl -s -X POST "https://rest.runpod.io/v1/endpoints" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"roof-corrosion-inference\",
        \"image\": \"${FULL_IMAGE}\",
        \"gpuTypeId\": \"NVIDIA_A10\",
        \"gpuCount\": 1,
        \"vcpuCount\": 4,
        \"memoryInGb\": 16,
        \"containerDiskInGb\": 10,
        \"env\": [
            {\"key\": \"MODELS_DIR\", \"value\": \"/app/models\"}
        ],
        \"networkVolumeMountPath\": \"/app/models\",
        \"minActiveWorkers\": 0,
        \"maxActiveWorkers\": 3,
        \"idleTimeout\": 300,
        \"flashBoot\": true
    }")

ENDPOINT_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', 'ERROR'))" 2>/dev/null || echo "ERROR")

if [ "${ENDPOINT_ID}" = "ERROR" ]; then
    echo "Failed to create endpoint. Response:"
    echo "${RESPONSE}"
    echo ""
    echo "You may already have an endpoint. Check with:"
    echo "  curl -s -H 'Authorization: Bearer \${RUNPOD_API_KEY}' https://rest.runpod.io/v1/endpoints | python3 -m json.tool"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Deployment Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "  Endpoint ID: ${ENDPOINT_ID}"
echo "  Image:       ${FULL_IMAGE}"
echo "  GPU:         NVIDIA A10 (24GB)"
echo ""
echo "  Add to .env.local:"
echo "    RUNPOD_ENDPOINT_ID=${ENDPOINT_ID}"
echo ""
echo "  Test with:"
echo "    curl -s -X POST \\"
echo "      https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \\"
echo "      -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": {\"image_url\": \"https://example.com/roof.jpg\", \"gsd\": 0.3}}'"
echo "═══════════════════════════════════════════════════════════"
