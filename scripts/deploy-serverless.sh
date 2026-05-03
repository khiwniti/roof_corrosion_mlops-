#!/usr/bin/env bash
# Deploy RunPod Serverless endpoint using template.json
#
# Usage: ./scripts/deploy-serverless.sh [NVIDIA_API_KEY]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_FILE="${REPO_ROOT}/infra/runpod/serverless/template.json"

# Load API key from .env if not provided
if [ -f "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env"
fi

NVIDIA_KEY="${1:-${NVIDIA_API_KEY:-}}"

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set. Add it to .env or export it."
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Deploying RunPod Serverless Template"
echo "  Template: ${TEMPLATE_FILE}"
echo "═══════════════════════════════════════════════════════════"

# Read and customize template
TEMPLATE_JSON=$(cat "${TEMPLATE_FILE}")

# Inject NVIDIA API key if provided
if [ -n "${NVIDIA_KEY}" ]; then
    echo "Using provided NVIDIA_API_KEY"
    TEMPLATE_JSON=$(echo "${TEMPLATE_JSON}" | sed "s/\"default\": \"\"/\"default\": \"${NVIDIA_KEY}\"/")
fi

# Create template via RunPod API
echo ""
echo "Creating serverless template..."
RESPONSE=$(curl -s -X POST "https://rest.runpod.io/v2/templates" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${TEMPLATE_JSON}" 2>/dev/null || echo '{"error":"curl failed"}')

# Check for errors
if echo "${RESPONSE}" | grep -q '"error"'; then
    echo "ERROR: Failed to create template"
    echo "Response: ${RESPONSE}"
    exit 1
fi

TEMPLATE_ID=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id','ERROR'))" 2>/dev/null || echo "ERROR")

if [ "${TEMPLATE_ID}" = "ERROR" ]; then
    echo "ERROR: Could not extract template ID"
    echo "Response: ${RESPONSE}"
    exit 1
fi

echo "✓ Template created: ${TEMPLATE_ID}"

# Create endpoint from template
echo ""
echo "Creating serverless endpoint..."
ENDPOINT_RESPONSE=$(curl -s -X POST "https://rest.runpod.io/v2/endpoints" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"roof-corrosion-inference\",
        \"templateId\": \"${TEMPLATE_ID}\",
        \"workersMin\": 0,
        \"workersMax\": 2,
        \"idleTimeout\": 300,
        \"flashboot\": true
    }" 2>/dev/null || echo '{"error":"curl failed"}')

if echo "${ENDPOINT_RESPONSE}" | grep -q '"error"'; then
    echo "ERROR: Failed to create endpoint"
    echo "Response: ${ENDPOINT_RESPONSE}"
    exit 1
fi

ENDPOINT_ID=$(echo "${ENDPOINT_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id','ERROR'))" 2>/dev/null || echo "ERROR")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Deployment Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "  Template ID:  ${TEMPLATE_ID}"
echo "  Endpoint ID:  ${ENDPOINT_ID}"
echo ""
echo "  Add to .env:"
echo "    RUNPOD_ENDPOINT_ID=${ENDPOINT_ID}"
echo ""
echo "  Test with:"
echo "    curl -s -X POST \\"
echo "      https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \\"
echo "      -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": {\"lat\": 13.7563, \"lng\": 100.5018, \"address\": \"Bangkok\"}}'"
echo "═══════════════════════════════════════════════════════════"

# Save to .env
if [ -f "${REPO_ROOT}/.env" ]; then
    if ! grep -q "RUNPOD_ENDPOINT_ID" "${REPO_ROOT}/.env"; then
        echo "RUNPOD_ENDPOINT_ID=${ENDPOINT_ID}" >> "${REPO_ROOT}/.env"
        echo "✓ Added RUNPOD_ENDPOINT_ID to .env"
    else
        sed -i '' "s/RUNPOD_ENDPOINT_ID=.*/RUNPOD_ENDPOINT_ID=${ENDPOINT_ID}/" "${REPO_ROOT}/.env" 2>/dev/null || \
        sed -i "s/RUNPOD_ENDPOINT_ID=.*/RUNPOD_ENDPOINT_ID=${ENDPOINT_ID}/" "${REPO_ROOT}/.env"
        echo "✓ Updated RUNPOD_ENDPOINT_ID in .env"
    fi
fi
