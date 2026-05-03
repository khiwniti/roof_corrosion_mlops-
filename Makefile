# Roof Corrosion MLOps — Makefile
# Common commands for development, training, and deployment

.PHONY: help install dev test train eval docker lint clean

# ── Help ─────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Install ──────────────────────────────────────────────────
install: ## Install all dependencies
	cd apps/web && pnpm install
	cd services/api && uv venv .venv --python 3.11 && . .venv/bin/activate && uv pip install -e ".[dev]"
	cd ml && uv venv .venv --python 3.11 && . .venv/bin/activate && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"

# ── Development ──────────────────────────────────────────────
dev-api: ## Start FastAPI dev server
	cd services/api && . .venv/bin/activate && uvicorn app.main:app --reload --port 8000

dev-web: ## Start Next.js dev server
	cd apps/web && pnpm dev

dev-worker: ## Start inference worker
	cd services/api && . .venv/bin/activate && python -m app.inference.worker --poll-interval 5

dev: ## Start all dev services (API + web + worker)
	@echo "Start each in separate terminals:"
	@echo "  make dev-api"
	@echo "  make dev-web"
	@echo "  make dev-worker"

# ── Docker ────────────────────────────────────────────────────
docker-up: ## Start local MLOps stack (MLflow, Redis, Label Studio, etc.)
	docker compose up -d
	@echo ""
	@echo "Services:"
	@echo "  MLflow:       http://localhost:5000"
	@echo "  Label Studio: http://localhost:8080"
	@echo "  Prefect:      http://localhost:4200"
	@echo "  MinIO:        http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  Grafana:      http://localhost:3001 (admin/admin)"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  API:          http://localhost:8000"

docker-down: ## Stop local MLOps stack
	docker compose down

docker-logs: ## Tail logs from all services
	docker compose logs -f

# ── Testing ──────────────────────────────────────────────────
test-api: ## Run API tests
	cd services/api && . .venv/bin/activate && pytest tests/ -v

test-web: ## Run web build check
	cd apps/web && pnpm build

test-licenses: ## Check data license compliance
	python3 ml/data/check_licenses.py

test: test-api test-web test-licenses ## Run all tests

# ── ML Training ──────────────────────────────────────────────
train-stage1: ## Train Stage 1 roof model (needs GPU)
	cd ml && . .venv/bin/activate && python ml/baseline/train_stage1_roof.py --config ml/baseline/configs/stage1_roof.yaml

train-stage2: ## Train Stage 2 corrosion model (needs GPU)
	cd ml && . .venv/bin/activate && python ml/baseline/train_stage2_corrosion.py --config ml/baseline/configs/stage2_corrosion.yaml

train-baseline: ## Run full Phase 1a baseline flow via Prefect
	cd ml && . .venv/bin/activate && python ml/flows/baseline_flow.py

# ── Evaluation ───────────────────────────────────────────────
eval-frozen: ## Evaluate model on frozen test set
	cd ml && . .venv/bin/activate && python ml/baseline/eval_frozen.py --model-uri $(MODEL_URI) --frozen-dir data/frozen_test

eval-drift: ## Run drift monitoring check
	cd ml && . .venv/bin/activate && python ml/eval/drift_monitor.py

# ── Data ─────────────────────────────────────────────────────
data-download: ## Download open datasets (interactive — requires accounts)
	bash ml/baseline/data/download_datasets.sh all

data-frozen-setup: ## Create frozen test set directory structure
	cd ml && . .venv/bin/activate && python ml/baseline/data/frozen_test_setup.py

# ── Lint ─────────────────────────────────────────────────────
lint-api: ## Lint API code
	cd services/api && . .venv/bin/activate && ruff check . && ruff format --check .

lint-ml: ## Lint ML code
	cd ml && . .venv/bin/activate && ruff check . && ruff format --check .

lint-web: ## Lint web code
	cd apps/web && pnpm lint

lint: lint-api lint-ml lint-web ## Lint everything

# ── Clean ─────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	rm -rf apps/web/.next services/api/.venv ml/.venv .turbo
