# Roof Corrosion MLOps — Project Rules

## Project Overview
Satellite-based roof corrosion detection → quote generation system.
- **Frontend**: Next.js 14 on Vercel (marketing + customer portal + ops dashboard)
- **API**: FastAPI on RunPod (two-stage segmentation inference)
- **ML**: Mask2Former (roof) + SegFormer (corrosion), trained on open data → synthetic → real
- **MLOps**: MLflow + Prefect + DVC + Label Studio + Evidently

## Key Commands

### Web (apps/web)
```bash
pnpm install          # install deps
pnpm dev              # dev server (port 3000)
pnpm build            # production build
pnpm lint             # eslint
pnpm typecheck        # tsc --noEmit
```

### API (services/api)
```bash
pip install -e ".[dev]"   # install with dev deps
uvicorn app.main:app --reload   # dev server (port 8000)
ruff check . && ruff format --check .   # lint
mypy app/                 # type check
pytest tests/ -v          # run tests
```

### ML (ml/)
```bash
pip install -e ".[dev]"   # install with dev deps
python ml/baseline/train_stage1_roof.py --config ml/baseline/configs/stage1_roof.yaml
python ml/baseline/train_stage2_corrosion.py --config ml/baseline/configs/stage2_corrosion.yaml
python ml/baseline/eval_frozen.py --model-uri <uri> --frozen-dir data/frozen_test
python ml/data/check_licenses.py   # CI gate: no NC data in prod
```

## Architecture Decisions
- **Two-stage model**: roof footprint (Mask2Former) → corrosion (SegFormer/UNet++). More debuggable than end-to-end.
- **Frozen real test set**: ~100–200 Maxar/Nearmap preview tiles, never used in training, never regenerated. All models bench against this.
- **Go/no-go gate (Phase 1a)**: Baseline on open data (Caribbean + AIRS + SpaceNet) must hit corrosion IoU ≥ 0.45 before investing in paid data.
- **DATA_LICENSES.md**: Every dataset tagged commercial-ok / research-only / NC. CI blocks NC data from prod models.
- **Confidence gating**: Quotes with low model confidence are forced to human review before sending.

## Data Licenses (critical)
- ✅ Commercial OK: Open AI Caribbean (CC-BY-4.0), SpaceNet (CC-BY-SA-4.0), MSFT/Google footprints
- ⚠️ Research only: AIRS, Inria — check terms before prod use
- ❌ NC (never in prod): xBD (CC-BY-NC-4.0), LandCover.ai (CC-BY-NC-SA)

## Production Deployment (Phase 3 — COMPLETE)

### RunPod Serverless Inference Endpoint
- **Endpoint ID**: `kpw4qpndjl9y55`
- **Template ID**: `0tmokvcftc`
- **Docker Image**: `docker.io/khiwnitigetintheq/roof-corrosion-inference:sha-b2b831e`
- **GPU**: AMPERE_16 (A10/16GB)
- **Workers**: min=0, max=2, idle_timeout=300s
- **API URL**: `https://api.runpod.ai/v2/kpw4qpndjl9y55/runsync`
- **Pipeline**: FM (OSM footprint + NVIDIA NIM VLM corrosion assessment)
- **Key Fix**: Handler must be `async def handler(job)` with `await _run(job)` — RunPod runs inside an existing event loop, so `asyncio.run()` causes "already running" crash

### API Service (FastAPI)
- **Docker Image**: `docker.io/khiwnitigetintheq/roof-corrosion-api:sha-b2b831e`
- **Lite Image**: No torch/rasterio/gdal — just FastAPI + httpx proxy
- **Serverless Proxy Mode**: When `RUNPOD_ENDPOINT_ID` is set, `/quote/sync` proxies to the inference endpoint instead of running local ML pipeline
- **Deployment**: Can run as RunPod Pod (with GPU) or any container host with the RUNPOD_ENDPOINT_ID env var

### Deploy Script
```bash
./scripts/deploy-production.sh [inference|api|all]
```

### Docker Image Cleanup
- Removed `dumb-init` from ENTRYPOINT (interferes with RunPod serverless runtime)
- Removed `HEALTHCHECK` from inference Dockerfile (not supported in serverless)
- All Dockerfiles use plain `CMD` for maximum compatibility

## Phased Roadmap
1. Phase 0: Monorepo + CI ✅
2. Phase 1a: Open-data baseline + go/no-go gate
3. Phase 1b: Paid data + synthetic + DA (gated on Phase 1a results)
4. Phase 2: Production training pipeline
5. Phase 3: RunPod serving ✅ **COMPLETE**
6. Phase 4: Vercel frontend
7. Phase 5: MLOps loop (drift, active learning, shadow deploys)
8. Phase 6: Quote engine + confidence gating
