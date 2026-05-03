# Roof Corrosion MLOps — Makefile
# Common commands for development, training, and deployment

.PHONY: help install dev test train eval docker lint clean \
       docker-build docker-push deploy deploy-inference deploy-training deploy-api

# ── Help ─────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Install ──────────────────────────────────────────────────
install: ## Install all dependencies
	cd apps/web && pnpm install
	cd services/api && uv venv .venv --python 3.11 && . .venv/bin/activate && uv pip install -e ".[dev]"
	cd ml && uv venv .venv --python 3.11 && . .venv/bin/activate && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[dev]"

# ── Development ──────────────────────────────────────────────
dev: ## Start all services (web + API + worker) via pnpm
	pnpm dev

dev-api: ## Start FastAPI dev server only
	bash scripts/dev-api.sh

dev-web: ## Start Next.js dev server only
	pnpm dev:web-only

dev-worker: ## Start inference worker only
	bash scripts/dev-worker.sh

dev-relabel: ## Start relabeling worker only
	bash scripts/dev-relabel-worker.sh

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

# ── Production Docker ─────────────────────────────────────────
DOCKER_USERNAME ?= khiwnitigetintheq
DOCKER_TAG ?= latest

docker-build-inference: ## Build inference Docker image (prod)
	docker buildx build --platform linux/amd64 \
		-f infra/runpod/serverless/inference/Dockerfile \
		-t docker.io/$(DOCKER_USERNAME)/roof-corrosion-inference:$(DOCKER_TAG) \
		.

docker-build-training: ## Build training Docker image (prod)
	docker buildx build --platform linux/amd64 \
		-f infra/runpod/serverless/training/Dockerfile \
		-t docker.io/$(DOCKER_USERNAME)/roof-corrosion-training:$(DOCKER_TAG) \
		.

docker-build-api: ## Build API Docker image (prod)
	docker buildx build --platform linux/amd64 \
		-f services/api/Dockerfile.runpod \
		-t docker.io/$(DOCKER_USERNAME)/roof-corrosion-api:$(DOCKER_TAG) \
		services/api

docker-build: docker-build-inference docker-build-training docker-build-api ## Build all production images

docker-push-inference: ## Push inference image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-inference:$(DOCKER_TAG)

docker-push-training: ## Push training image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-training:$(DOCKER_TAG)

docker-push-api: ## Push API image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-api:$(DOCKER_TAG)

docker-push: docker-push-inference docker-push-training docker-push-api ## Push all images

# ── Production Deploy ─────────────────────────────────────────
deploy-inference: ## Deploy inference to RunPod Serverless
	bash scripts/deploy-production.sh inference

deploy-training: ## Deploy training to RunPod Serverless
	bash scripts/deploy-production.sh training

deploy-api: ## Deploy API as RunPod pod
	bash scripts/deploy-production.sh api

deploy: ## Deploy all services to production
	bash scripts/deploy-production.sh all
