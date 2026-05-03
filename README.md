# Roof Corrosion AI — Satellite Roof Analysis & Quoting

Production MLOps system that turns a customer's address into a roof-replacement/repair quote derived from high-resolution satellite imagery of corrosion.

## Architecture

```
Vercel (Next.js 14)          RunPod (FastAPI + GPU inference)
┌──────────────────┐         ┌─────────────────────────────┐
│ / (marketing)    │         │ FastAPI gateway             │
│ /portal (quote)  │──JWT──▶ │  ├─ /quote  → Redis queue   │
│ /ops (dashboard) │         │  ├─ /feedback → relabel     │
└───────┬──────────┘         │  └─ /metrics (Prometheus)   │
        │                    │                              │
   Supabase Auth+DB          │  Inference Worker (A10/A100) │
   (customers, jobs,         │  ├─ Stage 1: Mask2Former    │
    quotes, feedback)        │  └─ Stage 2: SegFormer      │
                             └─────────────────────────────┘
                                      │
                             MLOps Control Plane
                             (MLflow, DVC, Prefect,
                              Label Studio, Evidently)
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+ / pnpm 9+
- Docker + Docker Compose (for local MLOps stack)
- GPU (for training; inference works on CPU but slowly)

### 1. Install dependencies

```bash
make install
```

### 2. Start local MLOps stack

```bash
make docker-up
```

This starts: MLflow (5000), Redis (6379), Label Studio (8080), MinIO (9001), Prefect (4200), Grafana (3001), Prometheus (9090).

### 3. Download open datasets

```bash
make data-download
```

Requires free accounts on DrivenData (Caribbean), GitHub (AIRS), and AWS (SpaceNet). See `DATA_LICENSES.md` for license details.

### 4. Train baseline models (Phase 1a)

```bash
make train-stage2    # Stage 2 (corrosion) — the go/no-go gate
make train-stage1    # Stage 1 (roof footprint)
```

**Go/no-go gate**: If corrosion IoU ≥ 0.45 on the frozen real test set → proceed to paid data. If < 0.25 → revisit task framing.

### 5. Run the API + worker

```bash
# Terminal 1: API
make dev-api

# Terminal 2: Inference worker
make dev-worker

# Terminal 3: Web frontend
make dev-web
```

### 6. Seed test data (optional)

```bash
python infra/supabase/seed.py
```

## Project Structure

```
├── apps/web/                    # Next.js 14 (Vercel)
│   └── src/app/
│       ├── page.tsx             # Marketing
│       ├── portal/page.tsx      # Customer quote portal
│       └── ops/page.tsx         # Internal MLOps dashboard
├── services/api/                # FastAPI (RunPod)
│   ├── app/routes/              # /quote, /feedback, /health, /metrics
│   ├── app/inference/           # Pipeline, worker, tile fetch
│   ├── app/db.py                # Supabase client + models
│   ├── app/queue.py             # Redis queue (with in-memory fallback)
│   └── app/quote_engine.py      # Assessment → itemized quote
├── ml/
│   ├── baseline/                # Phase 1a open-data baseline
│   │   ├── models/              # SegFormer-B3 (roof), SegFormer-B2 (corrosion)
│   │   ├── data/                # Dataset loaders (Caribbean, AIRS, SpaceNet, xBD)
│   │   ├── augmentation.py      # Albumentations pipelines
│   │   ├── train_stage1_roof.py
│   │   ├── train_stage2_corrosion.py
│   │   └── eval_frozen.py       # Immutable frozen test set evaluation
│   ├── eval/
│   │   ├── drift_monitor.py     # Evidently drift detection
│   │   └── active_learning.py   # Uncertainty + margin + diversity sampling
│   └── flows/
│       └── baseline_flow.py     # Prefect orchestration
├── infra/
│   ├── supabase/migrations/     # SQL schema (7 tables + RLS)
│   ├── runpod/                  # Pod configs (API, training, MLOps)
│   └── prometheus/              # Prometheus config
├── docker-compose.yml           # Local MLOps stack
├── DATA_LICENSES.md             # Dataset license tracking
├── AGENTS.md                    # Project rules + commands
└── Makefile                     # Common commands
```

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Imagery | Maxar 30–50cm | Only resolution that can see corrosion on residential roofs |
| Model | Two-stage (roof → corrosion) | More debuggable; matches how inspectors think |
| Stage 1 | SegFormer-B3 | Proven on overhead imagery; easier to train than Mask2Former |
| Stage 2 | SegFormer-B2 + Focal/Dice | Handles severe class imbalance (corrosion is rare) |
| Data | Open baseline → synthetic → real | De-risks with free data before spending on Maxar tiles |
| Frontend | Next.js 14 on Vercel | ISR for marketing, SSR for portal, free tier generous |
| API | FastAPI on RunPod | GPU inference; serverless for burst, persistent for baseline |
| DB/Auth | Supabase | RLS, Auth, Postgres in one; free tier for dev |
| Queue | Redis | Simple; in-memory fallback for local dev |
| MLOps | MLflow + Prefect + DVC + Evidently | Self-hosted, no vendor lock-in |

## Data Licenses

| Dataset | License | Commercial? |
|---|---|---|
| Open AI Caribbean | CC-BY-4.0 | ✅ Yes |
| SpaceNet | CC-BY-SA-4.0 | ✅ Yes (attribution) |
| AIRS | Research-only | ⚠️ Check |
| xBD (xView2) | CC-BY-NC-4.0 | ❌ No (research only) |

CI enforces: no NC-licensed data in production model weights. See `DATA_LICENSES.md`.

## Phased Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Monorepo + CI bootstrap |
| 1a | ✅ Scaffold | Open-data baseline + go/no-go gate |
| 1b | Pending | Maxar tile purchase + Label Studio + synthetic |
| 2 | Pending | Production training pipeline |
| 3 | Pending | RunPod serving |
| 4 | Pending | Vercel frontend (full) |
| 5 | Pending | MLOps loop (drift, active learning, shadow deploys) |
| 6 | Pending | Quote engine + confidence gating + PDF |

## Commands

```bash
make help          # Show all commands
make dev-api       # Start FastAPI on :8000
make dev-web       # Start Next.js on :3000
make dev-worker    # Start inference worker
make docker-up     # Start MLOps stack
make test          # Run all tests
make train-stage2  # Train corrosion model
make eval-frozen   # Evaluate on frozen test set
make eval-drift    # Run drift monitoring
make lint          # Lint all code
```
