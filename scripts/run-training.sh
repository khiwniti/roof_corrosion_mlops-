#!/usr/bin/env bash
# Trigger model training on RunPod Serverless
#
# Usage:
#   ./scripts/run-training.sh STAGE [EPOCHS] [DATA_SOURCE]
#
# Examples:
#   ./scripts/run-training.sh 1                     # Stage 1, 50 epochs, synthetic data
#   ./scripts/run-training.sh 2 80 synthetic        # Stage 2, 80 epochs, synthetic data
#   ./scripts/run-training.sh 1 50 s3               # Stage 1, 50 epochs, from S3

set -euo pipefail

STAGE="${1:?Usage: run-training.sh STAGE [EPOCHS] [DATA_SOURCE]}"
EPOCHS="${2:-50}"
DATA_SOURCE="${3:-synthetic}"

# Load config
if [ -f .env ]; then
    source .env
elif [ -f .env.local ]; then
    source .env.local
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set"
    exit 1
fi

if [ -z "${RUNPOD_TRAINING_ENDPOINT_ID:-}" ]; then
    echo "ERROR: RUNPOD_TRAINING_ENDPOINT_ID not set. Deploy training endpoint first:"
    echo "  ./scripts/deploy-training.sh"
    exit 1
fi

ENDPOINT_ID="${RUNPOD_TRAINING_ENDPOINT_ID}"
BACKBONE="b3"
S3_OUTPUT=""

if [ "${STAGE}" = "2" ]; then
    BACKBONE="b2"
fi

if [ -n "${AWS_S3_BUCKET:-}" ]; then
    S3_OUTPUT="s3://${AWS_S3_BUCKET}/models/"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Triggering Stage ${STAGE} Training on RunPod"
echo "  Epochs: ${EPOCHS}, Data: ${DATA_SOURCE}, Backbone: ${BACKBONE}"
echo "═══════════════════════════════════════════════════════════"

# Submit async training job
RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
        \"input\": {
            \"stage\": ${STAGE},
            \"epochs\": ${EPOCHS},
            \"batch_size\": 4,
            \"lr\": 0.0001,
            \"backbone\": \"${BACKBONE}\",
            \"data_source\": \"${DATA_SOURCE}\",
            \"upload_artifacts\": true,
            \"s3_output_path\": \"${S3_OUTPUT}\"
        }
    }")

JOB_ID=$(echo "${RESPONSE}" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r.get('id', r.get('error', 'UNKNOWN')))" 2>/dev/null)

if [ "${JOB_ID}" = "UNKNOWN" ] || [ -z "${JOB_ID}" ]; then
    echo "Failed to submit training job:"
    echo "${RESPONSE}"
    exit 1
fi

echo ""
echo "  Job submitted! ID: ${JOB_ID}"
echo ""
echo "  Check status:"
echo "    curl -s -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
echo "      https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID} | python3 -m json.tool"
echo ""
echo "  Or poll for completion:"
echo "    while true; do"
echo "      STATUS=\$(curl -s -H 'Authorization: Bearer \${RUNPOD_API_KEY}' \\"
echo "        https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID} | python3 -c 'import sys,json; print(json.load(sys.stdin).get(\"status\",\"?\"))')"
echo "      echo \"Status: \$STATUS\""
echo "      [ \"\$STATUS\" = \"COMPLETED\" ] || [ \"\$STATUS\" = \"FAILED\" ] && break"
echo "      sleep 30"
echo "    done"
echo "═══════════════════════════════════════════════════════════"
