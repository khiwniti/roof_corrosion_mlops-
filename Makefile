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
	uv sync --all-packages --all-extras

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
	uv run --package roof-corrosion-api pytest services/api/tests/ -v

test-ml: ## Run ML tests
	uv run --package roof-corrosion-ml pytest ml/tests/ -v --tb=short

test-web: ## Run web build check
	cd apps/web && pnpm build

test-licenses: ## Check data license compliance
	uv run python ml/data/check_licenses.py

test: test-api test-ml test-web test-licenses ## Run all tests

# ── ML Training ──────────────────────────────────────────────
train-stage1: ## Train Stage 1 roof model (needs GPU)
	uv run --package roof-corrosion-ml python ml/baseline/train_stage1_roof.py --config ml/baseline/configs/stage1_roof.yaml

train-stage2: ## Train Stage 2 corrosion model (needs GPU)
	uv run --package roof-corrosion-ml python ml/baseline/train_stage2_corrosion.py --config ml/baseline/configs/stage2_corrosion.yaml

train-baseline: ## Run full Phase 1a baseline flow via Prefect
	uv run --package roof-corrosion-ml python ml/flows/baseline_flow.py

train-multitask: ## Train Clay + Mask2Former multi-task model (Phase 2)
	uv run --package roof-corrosion-ml python ml/train/train_multitask.py --config ml/train/configs/clay_multitask.yaml

retrain-hitl: ## Run HITL retrain flow (Phase 4/5)
	uv run --package roof-corrosion-ml python ml/flows/retrain_flow.py

# ── Evaluation ───────────────────────────────────────────────
eval-frozen: ## Evaluate model on frozen test set
	uv run --package roof-corrosion-ml python ml/baseline/eval_frozen.py --model-uri $(MODEL_URI) --frozen-dir data/frozen_test

eval-drift: ## Run drift monitoring check
	uv run --package roof-corrosion-ml python ml/eval/drift_monitor.py

eval-seasonal: ## Run seasonal drift detection
	uv run --package roof-corrosion-ml python ml/eval/seasonal_drift.py

eval-cml: ## Run CML PR evaluation
	uv run --package roof-corrosion-ml python ml/eval/cml_pr_eval.py --output-dir reports/cml

eval-nannyml: ## Run NannyML label-free performance estimation
	uv run --package roof-corrosion-ml python ml/eval/nannyml_estimator.py

# ── Data ─────────────────────────────────────────────────────
data-download: ## Download open datasets (interactive — requires accounts)
	bash ml/baseline/data/download_datasets.sh all

data-frozen-setup: ## Create frozen test set directory structure
	uv run --package roof-corrosion-ml python ml/baseline/data/frozen_test_setup.py

# ── Lint ─────────────────────────────────────────────────────
lint-api: ## Lint API code
	uv run --package roof-corrosion-api ruff check services/api/ && uv run --package roof-corrosion-api ruff format --check services/api/

lint-ml: ## Lint ML code
	uv run --package roof-corrosion-ml ruff check ml/ && uv run --package roof-corrosion-ml ruff format --check ml/

lint-web: ## Lint web code
	cd apps/web && pnpm lint

lint: lint-api lint-ml lint-web ## Lint everything

# ── Clean ─────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name node_modules -exec rm -rf {} + 2>/dev/null || true
	rm -rf apps/web/.next .venv .turbo uv.lock

# ── Production Docker ─────────────────────────────────────────
DOCKER_USERNAME ?= khiwnitigetintheq
DOCKER_TAG ?= latest

docker-build-tiered: ## Build tiered handler Docker image (prod)
	docker buildx build --platform linux/amd64 \
		-f infra/runpod/serverless/tiered/Dockerfile \
		-t docker.io/$(DOCKER_USERNAME)/roof-corrosion-tiered:$(DOCKER_TAG) \
		.

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

docker-build: docker-build-tiered docker-build-inference docker-build-training docker-build-api ## Build all production images

docker-push-tiered: ## Push tiered image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-tiered:$(DOCKER_TAG)

docker-push-inference: ## Push inference image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-inference:$(DOCKER_TAG)

docker-push-training: ## Push training image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-training:$(DOCKER_TAG)

docker-push-api: ## Push API image to registry
	docker push docker.io/$(DOCKER_USERNAME)/roof-corrosion-api:$(DOCKER_TAG)

docker-push: docker-push-tiered docker-push-inference docker-push-training docker-push-api ## Push all images

# ── Production Deploy ─────────────────────────────────────────
deploy-tiered: ## Deploy tiered handler to RunPod Serverless
	bash scripts/deploy-production.sh tiered

deploy-inference: ## Deploy inference to RunPod Serverless
	bash scripts/deploy-production.sh inference

deploy-training: ## Deploy training to RunPod Serverless
	bash scripts/deploy-production.sh training

deploy-api: ## Deploy API as RunPod pod
	bash scripts/deploy-production.sh api

deploy: ## Deploy all services to production
	bash scripts/deploy-production.sh all
