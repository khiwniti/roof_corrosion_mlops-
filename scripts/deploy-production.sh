#!/usr/bin/env bash
# deploy-production.sh — Production deployment for Roof Corrosion MLOps
#
# Deploys:
#   1. Inference endpoint (RunPod Serverless)
#   2. API service (FastAPI — can be RunPod Pod or external)
#
# Prerequisites:
#   - RUNPOD_API_KEY env var set
#   - NVIDIA_API_KEY env var set
#   - Docker Hub credentials configured (for image push)
#   - runpod Python package installed
#
# Usage:
#   ./scripts/deploy-production.sh [inference|api|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEPLOY_TARGET="${1:-all}"
SHA_TAG="sha-$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD)"

echo "=== Roof Corrosion MLOps — Production Deployment ==="
echo "SHA tag: $SHA_TAG"
echo "Target: $DEPLOY_TARGET"
echo ""

# ── Build & Push Docker Images ────────────────────────────────────────────
build_and_push() {
    local name="$1"
    local dockerfile="$2"
    local context="$3"

    echo "Building $name image..."
    sudo docker build --platform linux/amd64 \
        -f "$dockerfile" \
        -t "docker.io/khiwnitigetintheq/roof-corrosion-${name}:latest" \
        -t "docker.io/khiwnitigetintheq/roof-corrosion-${name}:${SHA_TAG}" \
        "$context"

    echo "Pushing $name image..."
    sudo docker push "docker.io/khiwnitigetintheq/roof-corrosion-${name}:latest"
    sudo docker push "docker.io/khiwnitigetintheq/roof-corrosion-${name}:${SHA_TAG}"
    echo "✓ $name image pushed"
}

if [[ "$DEPLOY_TARGET" == "inference" || "$DEPLOY_TARGET" == "all" ]]; then
    echo "--- Inference Endpoint ---"
    build_and_push "inference" \
        "infra/runpod/serverless/inference/Dockerfile" \
        "$PROJECT_ROOT"

    echo "Updating RunPod Serverless template..."
    python3 -c "
import runpod, json, os
runpod.api_key = os.environ['RUNPOD_API_KEY']
# Template updates happen via the RunPod dashboard or API
# The inference endpoint (kpw4qpndjl9y55) uses the latest image
print('Inference endpoint: kpw4qpndjl9y55')
print('Template: tuycyp73h8')
"
    echo "✓ Inference endpoint ready"
fi

if [[ "$DEPLOY_TARGET" == "api" || "$DEPLOY_TARGET" == "all" ]]; then
    echo "--- API Service ---"
    build_and_push "api" \
        "services/api/Dockerfile.runpod-lite" \
        "$PROJECT_ROOT"

    echo "API image ready for deployment"
    echo "  - RunPod Pod: docker.io/khiwnitigetintheq/roof-corrosion-api:${SHA_TAG}"
    echo "  - Set RUNPOD_ENDPOINT_ID to the inference endpoint ID"
    echo "✓ API image pushed"
fi

echo ""
echo "=== Deployment Complete ==="
echo "Inference endpoint: https://api.runpod.ai/v2/kpw4qpndjl9y55/runsync"
echo "API image: docker.io/khiwnitigetintheq/roof-corrosion-api:${SHA_TAG}"
