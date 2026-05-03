#!/usr/bin/env bash
# Start FastAPI dev server with virtual environment activation
# Usage: bash scripts/dev-api.sh [--port 8000]

set -euo pipefail

PORT="${1:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/services/api/.venv"

# Activate venv
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  No venv found at $VENV_DIR"
    echo "   Run: cd services/api && uv venv .venv --python 3.11 && uv pip install -e '.[dev]'"
    exit 1
fi

echo "Starting FastAPI on :$PORT..."
uvicorn app.main:app --reload --port "$PORT" --app-dir "$PROJECT_DIR/services/api"
