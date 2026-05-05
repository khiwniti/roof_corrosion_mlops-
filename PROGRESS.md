# Implementation Progress Tracker

> Last updated: May 2026

## Phase 0 — Foundations

| Component | Status | Notes |
|-----------|--------|-------|
| Monorepo (Turborepo + pnpm workspace) | ✅ Complete | apps/web, services/api, ml/, infra/ |
| Next.js 15 + MapLibre GL 5 + terra-draw | ✅ Complete | Portal with polygon drawing, real-time job status |
| Supabase (Postgres + Realtime) | ✅ Complete | Jobs table, webhook updates, auth stub |
| Cloudflare R2 | ✅ Complete | Mask uploads, pre-signed URLs |
| RunPod serverless handler stub | ✅ Complete | Tier-0 S2 pipeline with fallback |
| CI/CD (GitHub Actions) | ✅ Complete | Web build, API tests, ML tests, CML eval |

## Phase 1a — Tier-0 Free Preliminary

| Component | Status | Notes |
|-----------|--------|-------|
| CDSE Sentinel-2 ingestion | 🟡 Stub | Real pipeline needs CDSE credentials |
| Feature extraction (indices, GLCM) | 🟡 Stub | Needs xarray dependency in CI |
| Rule-based material classifier | ✅ Complete | 5-class + corrosion stub |
| Area estimation | ✅ Complete | GSD-based pixel-to-m² conversion |
| Mask generation + R2 upload | ✅ Complete | PNG + base64 fallback |
| Webhook payload | ✅ Complete | Tier-0 metadata with corrosion |
| Portal UI (tier selector, quote) | ✅ Complete | Dynamic quotes, THB 100k threshold |

## Phase 2 — Labeling + Foundation Model

| Component | Status | Notes |
|-----------|--------|-------|
| Clay v1.5 + ViT-Adapter + Mask2Former | 🟡 Stub | Architecture defined, training script scaffolded |
| Multi-task model (3 heads) | 🟡 Stub | material, corrosion, severity |
| Training pipeline (TerraTorch) | 🟡 Stub | Config + dataset class ready |
| MLflow logging | 🟡 Stub | Registry module ready |

## Phase 3 — Tier-1 Paid VHR

| Component | Status | Notes |
|-----------|--------|-------|
| Airbus OneAtlas API client | ✅ Complete | Search + cost estimation |
| GISTDA THEOS-2 API client | 🟡 Stub | Search + cost estimation placeholders |
| Building detector (YOLO/Mask R-CNN) | 🟡 Stub | Random bbox generation |
| SAM2 roof segmentation | 🟡 Stub | Random mask generation |
| Clay multi-task inference | 🟡 Stub | Temperature scaling included |
| Handler Tier-1 routing | ✅ Complete | Fallback to Tier-0 if no VHR |
| Portal tier selector | ✅ Complete | Tier badge, cost estimates |
| RunPod tiered template | ✅ Complete | Dockerfile + template.json |

## Phase 4/5 — HITL Flywheel + MLOps

| Component | Status | Notes |
|-----------|--------|-------|
| HITL retrain Prefect flow | ✅ Complete | Active learning, evaluation, promotion |
| Model registry (MLflow/S3/local) | ✅ Complete | load/save/list checkpoints |
| Evidently drift monitoring | ✅ Complete | Input + prediction drift reports |
| alibi-detect seasonal drift | ✅ Complete | ContextMMDDrift stub |
| NannyML label-free performance | ✅ Complete | CBPE + DLE stubs |
| CML PR evaluation | ✅ Complete | Markdown report generation, CI integration |
| BatchBALD prioritization | 🟡 Stub | Random selection placeholder |
| Label Studio integration | 🟡 Stub | Queue update placeholder |

## Phase 6 — Quote Engine

| Component | Status | Notes |
|-----------|--------|-------|
| Dynamic quote generation | ✅ Complete | Severity-based line items |
| Confidence gating (THB 100k) | ✅ Complete | Mandatory human review flag |
| Temperature scaling | ✅ Complete | Calibrated confidence per province |
| 7-year retention | ❌ Not started | Infrastructure TBD |

## Test Coverage

- **ML tests**: 26 passing (2 skipped: xarray, prefect)
- **Web build**: ✅ Passing
- **Handler**: ✅ Runs locally with stub fallbacks

## Next Milestones

1. **Real model training**: Requires GPU compute + labeled dataset
2. **CDSE integration**: Needs Copernicus account + S3 credentials
3. **Airbus OneAtlas subscription**: €3.80/km² for archive imagery
4. **VHR ground truth**: 100–200 manually labeled Pléiades tiles for frozen test set
5. **Drone partner onboarding**: CAAT-licensed pilots in 5 Thai provinces
