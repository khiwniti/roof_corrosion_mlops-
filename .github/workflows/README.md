# CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 1. `ci.yml` — Continuous Integration
**Triggers**: Push to `main`/`develop`, PR to `main`

Runs on every commit:
- **Web (Next.js)**: Lint, typecheck, build
- **API (FastAPI)**: Ruff lint/format, mypy typecheck, pytest + coverage
- **ML**: Ruff lint/format, data license check
- **License Gate**: Validates `DATA_LICENSES.md` compliance

### 2. `cd.yml` — Continuous Deployment
**Triggers**: Push to `main`, tags `v*`, manual dispatch

Automated deployment pipeline:
1. Build Docker image (`linux/amd64` for RunPod)
2. Push to Docker Hub with tags: `latest`, `sha-<short>`, semver
3. Deploy to RunPod Serverless endpoint

### 3. `pr-docker.yml` — PR Docker Validation
**Triggers**: PR modifying Docker or API code

Builds Docker image without pushing to validate:
- Dockerfile syntax
- Image builds successfully
- Python imports work inside container

### 4. `deploy-web.yml` — Frontend Deployment
**Triggers**: Push to `main` modifying web code

Deploys Next.js app to Vercel.

## Required Secrets & Variables

### Docker Hub (for CD)
Go to **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Docker Hub username (`khiwnitigetintheq`) |
| `DOCKER_PASSWORD` | Docker Hub access token (not password) |

### RunPod (for CD)
| Secret | Description |
|--------|-------------|
| `RUNPOD_API_KEY` | Your RunPod API key |
| `RUNPOD_ENDPOINT_ID` | The endpoint ID to update (set as Variable, not Secret) |

### Vercel (for web deploy)
| Secret | Description |
|--------|-------------|
| `VERCEL_TOKEN` | Vercel personal access token |
| `VERCEL_ORG_ID` | Vercel organization ID |
| `VERCEL_PROJECT_ID` | Vercel project ID |

### Repository Variables
| Variable | Example | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `https://api.runpod.ai/v2/xxx` | RunPod endpoint URL for frontend |
| `NEXT_PUBLIC_REGION` | `TH` | Default region for quotes |

## Setting up Secrets

1. Go to https://github.com/khiwniti/roof_corrosion_mlops-/settings/secrets/actions
2. Click **New repository secret**
3. Add each secret from the table above

## Setting up Variables

1. Go to https://github.com/khiwniti/roof_corrosion_mlops-/settings/variables/actions
2. Click **New repository variable**
3. Add `RUNPOD_ENDPOINT_ID`, `NEXT_PUBLIC_API_URL`, etc.

## Testing Workflows Locally

### Docker build
```bash
docker build --platform linux/amd64 \
  -f infra/runpod/serverless/inference/Dockerfile \
  -t roof-corrosion-inference:test .
```

### API tests
```bash
cd services/api
pip install -e ".[dev]"
pytest tests/ -v
```

### Web build
```bash
cd apps/web
pnpm install
pnpm build
```
