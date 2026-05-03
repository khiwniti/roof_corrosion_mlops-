#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Roof Corrosion MLOps — Production Deployment Script
#
# Builds Docker images, pushes to Docker Hub, and deploys to RunPod Serverless.
#
# Usage (from repo root):
#   ./scripts/deploy-production.sh inference          # deploy inference only
#   ./scripts/deploy-production.sh training           # deploy training only
#   ./scripts/deploy-production.sh api                # deploy API only
#   ./scripts/deploy-production.sh all                # deploy everything
#   ./scripts/deploy-production.sh all --no-push      # build only, don't push
#
# Environment:
#   DOCKER_USERNAME  — Docker Hub username (or set in .env)
#   DOCKER_PASSWORD  — Docker Hub password/token
#   RUNPOD_API_KEY   — RunPod API key for endpoint creation
#   NVIDIA_API_KEY   — NVIDIA NIM API key (injected into inference endpoint)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Configuration ─────────────────────────────────────────────────────────
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
USERNAME="${DOCKER_USERNAME:-khiwnitigetintheq}"
TAG="${DEPLOY_TAG:-latest}"
PLATFORM="linux/amd64"
PUSH=true
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
TARGET="${1:-all}"
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-push) PUSH=false; shift ;;
        --tag=*) TAG="${1#*=}"; shift ;;
        --username=*) USERNAME="${1#*=}"; shift ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Image names
INFERENCE_IMAGE="${REGISTRY}/${USERNAME}/roof-corrosion-inference:${TAG}"
TRAINING_IMAGE="${REGISTRY}/${USERNAME}/roof-corrosion-training:${TAG}"
API_IMAGE="${REGISTRY}/${USERNAME}/roof-corrosion-api:${TAG}"

# ── Load env ──────────────────────────────────────────────────────────────
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
elif [ -f "${REPO_ROOT}/.env.local" ]; then
    set -a; source "${REPO_ROOT}/.env.local"; set +a
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Roof Corrosion MLOps — Production Deployment"
echo "═══════════════════════════════════════════════════════════════"
echo "  Target:   ${TARGET}"
echo "  Registry: ${REGISTRY}/${USERNAME}"
echo "  Tag:      ${TAG}"
echo "  Push:     ${PUSH}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Docker login ──────────────────────────────────────────────────────────
if [ "$PUSH" = true ]; then
    if [ -z "${DOCKER_USERNAME:-}" ] || [ -z "${DOCKER_PASSWORD:-}" ]; then
        err "DOCKER_USERNAME and DOCKER_PASSWORD must be set for push"
        err "Set them in .env, .env.local, or as environment variables"
        exit 1
    fi
    info "Logging in to Docker Hub..."
    echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin "${REGISTRY}"
    ok "Docker login successful"
fi

# ── Build functions ───────────────────────────────────────────────────────
build_inference() {
    info "Building inference image: ${INFERENCE_IMAGE}"
    docker buildx build --platform "${PLATFORM}" \
        -f "${REPO_ROOT}/infra/runpod/serverless/inference/Dockerfile" \
        -t "${INFERENCE_IMAGE}" \
        --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${REPO_ROOT}"
    ok "Inference image built"
}

build_training() {
    info "Building training image: ${TRAINING_IMAGE}"
    docker buildx build --platform "${PLATFORM}" \
        -f "${REPO_ROOT}/infra/runpod/serverless/training/Dockerfile" \
        -t "${TRAINING_IMAGE}" \
        --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${REPO_ROOT}"
    ok "Training image built"
}

build_api() {
    info "Building API image: ${API_IMAGE}"
    docker buildx build --platform "${PLATFORM}" \
        -f "${REPO_ROOT}/services/api/Dockerfile.runpod" \
        -t "${API_IMAGE}" \
        --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${REPO_ROOT}/services/api"
    ok "API image built"
}

# ── Push functions ────────────────────────────────────────────────────────
push_image() {
    local image="$1"
    if [ "$PUSH" = true ]; then
        info "Pushing: ${image}"
        docker push "${image}"
        ok "Pushed: ${image}"
    else
        warn "Skipping push (--no-push): ${image}"
    fi
}

# ── RunPod deployment functions ───────────────────────────────────────────
deploy_inference_endpoint() {
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        warn "RUNPOD_API_KEY not set — skipping endpoint deployment"
        return
    fi

    info "Creating/updating RunPod inference endpoint..."

    RESPONSE=$(curl -sf -X POST "https://rest.runpod.io/v1/endpoints" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"roof-corrosion-inference\",
            \"image\": \"${INFERENCE_IMAGE}\",
            \"gpuTypeId\": \"NVIDIA RTX A4000\",
            \"gpuCount\": 1,
            \"vcpuCount\": 4,
            \"memoryInGb\": 16,
            \"containerDiskInGb\": 10,
            \"env\": [
                {\"key\": \"REGION\", \"value\": \"${REGION:-TH}\"},
                {\"key\": \"PIPELINE\", \"value\": \"fm\"},
                {\"key\": \"PYTHONPATH\", \"value\": \"/app/src\"},
                {\"key\": \"PYTHONUNBUFFERED\", \"value\": \"1\"},
                {\"key\": \"MIN_CONFIDENCE\", \"value\": \"0.6\"},
                {\"key\": \"LOG_LEVEL\", \"value\": \"INFO\"},
                {\"key\": \"NVIDIA_API_KEY\", \"value\": \"${NVIDIA_API_KEY:-}\"}
            ],
            \"minActiveWorkers\": 0,
            \"maxActiveWorkers\": 3,
            \"idleTimeout\": 300,
            \"flashBoot\": true
        }" 2>/dev/null || echo '{"error":"curl failed"}')

    ENDPOINT_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', 'ERROR'))" 2>/dev/null || echo "ERROR")

    if [ "${ENDPOINT_ID}" = "ERROR" ]; then
        warn "Could not create endpoint (may already exist). Response: ${RESPONSE}"
    else
        ok "Inference endpoint: ${ENDPOINT_ID}"
        echo ""
        echo "  Add to .env.local:"
        echo "    RUNPOD_INFERENCE_ENDPOINT_ID=${ENDPOINT_ID}"
    fi
}

deploy_training_endpoint() {
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        warn "RUNPOD_API_KEY not set — skipping endpoint deployment"
        return
    fi

    info "Creating/updating RunPod training endpoint..."

    RESPONSE=$(curl -sf -X POST "https://rest.runpod.io/v1/endpoints" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"roof-corrosion-training\",
            \"image\": \"${TRAINING_IMAGE}\",
            \"gpuTypeId\": \"NVIDIA_A100_80GB\",
            \"gpuCount\": 1,
            \"vcpuCount\": 8,
            \"memoryInGb\": 32,
            \"containerDiskInGb\": 50,
            \"env\": [
                {\"key\": \"MODELS_DIR\", \"value\": \"/app/models\"},
                {\"key\": \"DATA_DIR\", \"value\": \"/app/data\"},
                {\"key\": \"PYTHONUNBUFFERED\", \"value\": \"1\"},
                {\"key\": \"LOG_LEVEL\", \"value\": \"INFO\"}
            ],
            \"minActiveWorkers\": 0,
            \"maxActiveWorkers\": 1,
            \"idleTimeout\": 300
        }" 2>/dev/null || echo '{"error":"curl failed"}')

    ENDPOINT_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', 'ERROR'))" 2>/dev/null || echo "ERROR")

    if [ "${ENDPOINT_ID}" = "ERROR" ]; then
        warn "Could not create endpoint (may already exist). Response: ${RESPONSE}"
    else
        ok "Training endpoint: ${ENDPOINT_ID}"
        echo ""
        echo "  Add to .env.local:"
        echo "    RUNPOD_TRAINING_ENDPOINT_ID=${ENDPOINT_ID}"
    fi
}

# ── Execute ───────────────────────────────────────────────────────────────
case "${TARGET}" in
    inference)
        build_inference
        push_image "${INFERENCE_IMAGE}"
        deploy_inference_endpoint
        ;;
    training)
        build_training
        push_image "${TRAINING_IMAGE}"
        deploy_training_endpoint
        ;;
    api)
        build_api
        push_image "${API_IMAGE}"
        ;;
    all)
        build_inference
        build_training
        build_api
        push_image "${INFERENCE_IMAGE}"
        push_image "${TRAINING_IMAGE}"
        push_image "${API_IMAGE}"
        deploy_inference_endpoint
        deploy_training_endpoint
        ;;
    *)
        err "Unknown target: ${TARGET}"
        err "Usage: $0 {inference|training|api|all} [--no-push] [--tag=X]"
        exit 1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Deployment Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Images:"
[ "${TARGET}" = "inference" ] || [ "${TARGET}" = "all" ] && echo "    Inference: ${INFERENCE_IMAGE}"
[ "${TARGET}" = "training" ] || [ "${TARGET}" = "all" ] && echo "    Training:  ${TRAINING_IMAGE}"
[ "${TARGET}" = "api" ] || [ "${TARGET}" = "all" ] && echo "    API:       ${API_IMAGE}"
echo ""
echo "  Test inference:"
echo "    curl -s -X POST \\"
echo "      https://api.runpod.ai/v2/\${RUNPOD_INFERENCE_ENDPOINT_ID}/runsync \\"
echo "      -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": {\"lat\": 13.7563, \"lng\": 100.5018, \"gsd\": 0.3}}'"
echo ""
echo "═══════════════════════════════════════════════════════════════"
