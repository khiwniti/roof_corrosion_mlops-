#!/usr/bin/env bash
# Start inference worker with virtual environment activation
# Usage: bash scripts/dev-worker.sh [--poll-interval 5]

set -euo pipefail

POLL_INTERVAL="${1:-5}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/services/api/.venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  No venv found at $VENV_DIR"
    exit 1
fi

echo "Starting inference worker (poll: ${POLL_INTERVAL}s)..."
cd "$PROJECT_DIR/services/api"
python -m app.inference.worker --poll-interval "$POLL_INTERVAL"
