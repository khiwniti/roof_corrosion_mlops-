# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Tier-1 VHR Pipeline (Phase 3)**
  - `ml/ingestion/pleiades.py` — Airbus OneAtlas API client for archive search and cost estimation
  - `ml/ingestion/theos2.py` — GISTDA THEOS-2 API client stub
  - `ml/inference/tier1.py` — Tier-1 inference stub with building detection, SAM2 segmentation, Clay classification, temperature scaling
  - Tiered RunPod handler routing with fallback to Tier-0
  - Portal tier selector UI with cost estimates and dynamic quote bands
  - `infra/runpod/serverless/tiered/template.json` for RunPod deployment
- **Corrosion Prediction**
  - Added corrosion probability and severity stubs to Tier-0 classifier
  - Portal displays corrosion % and severity with color-coded badges
  - Quote generation uses actual corrosion data from inference metadata
- **Model Registry**
  - `ml/train/models/registry.py` — Checkpoint loading/saving for MLflow, S3/R2, and local filesystem
  - Supports `load_model()`, `load_checkpoint()`, `save_checkpoint()`, `list_local_checkpoints()`
- **MLOps Evaluation (Phase 5)**
  - `ml/eval/seasonal_drift.py` — alibi-detect ContextMMDDrift stub for tropical seasonal monitoring
  - `ml/eval/nannyml_estimator.py` — NannyML CBPE/DLE label-free performance estimation
  - `ml/eval/cml_pr_eval.py` — CML PR evaluation pipeline with markdown report generation
  - CI integration for CML evaluation on pull requests
- **HITL Retrain Flow (Phase 4)**
  - `ml/flows/retrain_flow.py` — Prefect flow for active learning, retraining, evaluation, and MLflow promotion
  - `ml/flows/baseline_flow.py` — Phase 1a baseline training orchestration
- **Testing**
  - 26 ML tests covering ingestion, inference, registry, drift, and evaluation
  - CI `ml-test` job in GitHub Actions
- **Documentation**
  - `PROGRESS.md` — Implementation status tracker for all phases
  - Updated `DEPLOY_GUIDE.md` with tiered handler deployment instructions
  - Updated `Makefile` with tiered Docker, ML tests, drift, CML, and retrain commands

### Changed
- `apps/web/src/app/portal/page.tsx` — Added tier selector, corrosion display, dynamic quote generation
- `apps/web/src/app/api/runpod/webhook/route.ts` — Stores corrosion, severity, and corroded_area_px metadata
- `infra/runpod/serverless/tiered/handler.py` — Tier-1 routing with VHR fallback, corrosion fields in webhook
- `.github/workflows/ci.yml` — Added `ml-test` and `ml-cml-eval` jobs
- `ml/inference/tier0.py` — Added corrosion_prob and severity to prediction output

### Fixed
- `ml/tests/test_tier0.py` — Updated to match actual `predict()` and `estimate_roof_area()` API
- `ml/tests/test_features.py` — Skips when xarray is not installed
- `ml/train/models/__init__.py` — Uses relative imports to avoid package path issues
- `datetime.utcnow()` deprecation across `ml/eval/` modules — replaced with `datetime.now(UTC)`

## [0.1.0] — 2025-XX-XX

### Added
- Initial monorepo setup with Next.js 15, FastAPI, and ML pipeline
- Tier-0 Sentinel-2 ingestion pipeline with CDSE STAC client
- Rule-based material classifier stub
- Portal UI with MapLibre polygon drawing
- RunPod serverless handler for GPU inference
- MLflow + Prefect + DVC MLOps stack
