# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 1. `ci.yml` — Continuous Integration
**Triggers**: Push to `main`/`develop`, PR to `main`

Runs on every commit:
- **Web (Next.js)**: Lint, typecheck, build
- **API (FastAPI)**: Ruff lint/format, mypy typecheck, pytest + coverage (via `uv`)
- **ML**: Ruff lint/format, data license check (via `uv`)
- **License Gate**: Validates `DATA_LICENSES.md` compliance

### 2. `cd.yml` — API Docker Build
**Triggers**: Push to `main` modifying `services/api/`, tags `v*`, manual dispatch

Builds and pushes the FastAPI API container (`services/api/Dockerfile.runpod`):
1. Build `linux/amd64` image using Docker Buildx + GHA cache
2. Push to Docker Hub with tags: `latest`, `sha-<short>`, semver

### 3. `deploy-runpod.yml` — RunPod Serverless Deployment
**Triggers**: Push to `main`, tags `v*`, manual dispatch (`inference` / `training` / `both`)

Deploys to RunPod Serverless:
1. Build inference image (`infra/runpod/serverless/inference/Dockerfile`)
2. Build training image (`infra/runpod/serverless/training/Dockerfile`) — tags and manual `both`
3. Update the inference endpoint via RunPod REST API + smoke test
4. Update the training endpoint via RunPod REST API

### 4. `pr-docker.yml` — PR Docker Validation
**Triggers**: PR modifying Docker or API code

Builds the inference image without pushing to validate:
- Dockerfile syntax and layer caching
- Image builds successfully with uv
- Python imports work inside container

### 5. `deploy-web.yml` — Frontend Deployment
**Triggers**: Push to `main` modifying web code

Deploys Next.js app to Vercel.

## Required Secrets & Variables

> **Without these secrets the build still runs — the image is built but not pushed, and RunPod deployment is skipped.** Add them when you're ready to ship.

Set these in **Settings → Secrets and variables → Actions**.

### Secrets

| Secret | Used by | Description |
|--------|---------|-------------|
| `DOCKER_USERNAME` | cd, deploy-runpod | Docker Hub username |
| `DOCKER_PASSWORD` | cd, deploy-runpod | Docker Hub access token |
| `RUNPOD_API_KEY` | deploy-runpod | RunPod API key |
| `VERCEL_TOKEN` | deploy-web | Vercel personal access token |
| `VERCEL_ORG_ID` | deploy-web | Vercel organization ID |
| `VERCEL_PROJECT_ID` | deploy-web | Vercel project ID |

### Variables (not secrets — safe to log)

| Variable | Example | Description |
|----------|---------|-------------|
| `DOCKER_USERNAME` | `khiwniti` | Docker Hub username (also as variable for image tags) |
| `RUNPOD_INFERENCE_ENDPOINT_ID` | `abc123def` | RunPod inference endpoint ID |
| `RUNPOD_TRAINING_ENDPOINT_ID` | `xyz789uvw` | RunPod training endpoint ID |
| `NEXT_PUBLIC_API_URL` | `https://api.runpod.ai/v2/xxx` | RunPod endpoint URL for frontend |
| `NEXT_PUBLIC_REGION` | `TH` | Default region for quotes |

## First-time RunPod Setup

1. Build and push the inference image manually:
   ```bash
   ./scripts/deploy-inference.sh docker.io/YOUR_USERNAME v1.0
   ```

2. This creates the RunPod endpoint and prints the `RUNPOD_INFERENCE_ENDPOINT_ID`.

3. Add that ID as a GitHub repository variable (`RUNPOD_INFERENCE_ENDPOINT_ID`).

4. Subsequent pushes to `main` will update the endpoint automatically.

5. For the training endpoint:
   ```bash
   ./scripts/deploy-training.sh docker.io/YOUR_USERNAME v1.0
   ```
   Add the printed endpoint ID as `RUNPOD_TRAINING_ENDPOINT_ID`.

## Testing Workflows Locally

### Inference Docker build
```bash
docker build --platform linux/amd64 \
  -f infra/runpod/serverless/inference/Dockerfile \
  -t roof-corrosion-inference:test .

docker run --rm \
  -e REGION=TH -e PYTHONPATH=/app/src \
  roof-corrosion-inference:test \
  python3 -c "import app.inference.pipeline_fm; print('OK')"
```

### Training Docker build
```bash
docker build --platform linux/amd64 \
  -f infra/runpod/serverless/training/Dockerfile \
  -t roof-corrosion-training:test .
```

### API tests (using uv)
```bash
cd services/api
uv pip install --system -e ".[dev]"
pytest tests/ -v
```

### Web build
```bash
cd apps/web
pnpm install
pnpm build
```
