#!/usr/bin/env bash
# Roof Corrosion MLOps — Development Environment Setup
#
# Checks prerequisites, installs dependencies, and verifies the setup.
# Run: bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found ($(command -v "$1"))"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

check_version() {
    local cmd="$1"
    local min_version="$2"
    local current_version

    current_version=$($cmd --version 2>&1 | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    if [ "$(printf '%s\n' "$min_version" "$current_version" | sort -V | head -n1)" = "$min_version" ]; then
        echo -e "${GREEN}✓${NC} $cmd version $current_version (>= $min_version)"
        return 0
    else
        echo -e "${RED}✗${NC} $cmd version $current_version (< $min_version)"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════════"
echo "  Roof Corrosion MLOps — Development Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Prerequisites ──────────────────────────────────────────

echo "Checking prerequisites..."
MISSING=0

check_command "python3" || MISSING=1
check_command "node" || MISSING=1
check_command "pnpm" || MISSING=1
check_command "docker" || MISSING=1
check_command "uv" || MISSING=1

check_version "python3" "3.11" || MISSING=1
check_version "node" "20.0" || MISSING=1

if [ $MISSING -ne 0 ]; then
    echo ""
    echo -e "${RED}Please install missing prerequisites and re-run.${NC}"
    echo "  Python 3.11+: https://python.org"
    echo "  Node.js 20+: https://nodejs.org"
    echo "  pnpm: https://pnpm.io/installation"
    echo "  Docker: https://docker.com"
    echo "  uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo ""
echo -e "${GREEN}All prerequisites met.${NC}"

# ── Install dependencies ───────────────────────────────────

echo ""
echo "Installing dependencies..."

cd "$REPO_ROOT"

echo "  → Python packages (uv sync)..."
uv sync --all-packages --all-extras

echo "  → Node.js packages (pnpm install)..."
cd apps/web
pnpm install

cd "$REPO_ROOT"

# ── Verify ─────────────────────────────────────────────────

echo ""
echo "Verifying setup..."

echo "  → ML tests"
uv run --package roof-corrosion-ml pytest ml/tests/ -q --tb=short || true

echo "  → Web build"
cd apps/web
pnpm build > /dev/null 2>&1 && echo -e "    ${GREEN}✓${NC} Web build passed" || echo -e "    ${YELLOW}!${NC} Web build had issues (check manually)"

cd "$REPO_ROOT"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Quick start:"
echo "  make docker-up    # Start local MLOps stack"
echo "  make dev          # Start web + API dev servers"
echo "  make test         # Run all tests"
echo "  make test-ml      # Run ML tests only"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in credentials"
echo "  2. Run: python scripts/test-handler-local.py"
echo "  3. Open: http://localhost:3000 (portal)"
echo ""
