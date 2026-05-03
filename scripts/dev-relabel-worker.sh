#!/usr/bin/env bash
# Start relabeling worker with virtual environment activation
# Usage: bash scripts/dev-relabel-worker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/services/api/.venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  No venv found at $VENV_DIR"
    exit 1
fi

echo "Starting relabeling worker..."
cd "$PROJECT_DIR/services/api"
python -m app.inference.relabel_worker --poll-interval 30
