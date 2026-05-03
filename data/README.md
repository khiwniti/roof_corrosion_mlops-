# Thailand Data Sources

This project is scoped to **Thailand** (`REGION=TH`) for the MVP.
We deliberately avoid bulk-downloading global datasets (SpaceNet, AIRS, xBD)
because the foundation-model pipeline (`PIPELINE=fm`) doesn't need them.

## What we use (no download required)

### 1. Building footprints
- **Primary**: OpenStreetMap Overpass API (real-time, free)
  - Thailand has very good OSM coverage in cities (Bangkok, Chiang Mai, Phuket, Hat Yai)
  - Rural coverage is patchy — we fall back to bbox or SAM 2
- **Backup**: Microsoft Global ML Building Footprints (Thailand subset on demand)
  - Static GeoJSON, ~50MB for Thailand vs ~9GB for SpaceNet global
  - Source: `https://github.com/microsoft/GlobalMLBuildingFootprints`

### 2. Satellite imagery (per request)
- **Maxar** Vivid (≤30cm GSD) — paid, but only fetches the tile for the queried address
- **Nearmap** PhotoMaps (≤10cm GSD) — paid, available in Bangkok metro
- **Google Static Maps** / **Mapbox Satellite** — easier to start with, ~50cm GSD
- **GISTDA THEOS-2** — Thailand's national satellite, ~50cm GSD, public-sector access

### 3. VLM (corrosion assessment)
- **NVIDIA NIM** Llama-3.2-90B-Vision (hosted at `integrate.api.nvidia.com`)
- Free tier: 1,000 credits at signup
- No image storage on NVIDIA side (per their TOS)

## What we explicitly do NOT use

| Dataset | Size | Why not |
|---|---|---|
| SpaceNet 7 (global buildings) | 9 GB | Replaced by OSM Overpass |
| AIRS (Christchurch, NZ) | 2 GB | Not relevant to Thailand |
| xBD (xView2 disaster imagery) | 30 GB | NC-licensed, not Thailand |
| Open AI Caribbean Challenge | 1 GB | Caribbean, not Thailand |
| LandCover.ai (Poland) | 5 GB | NC-licensed, not Thailand |

## If you want to train a Thailand-specific model

Optional fine-tuning data sources for the trained SegFormer pipeline:

1. **Manual curation**: Use the FM pipeline to label ~500 Thai roofs, then
   correct in Label Studio → train SegFormer-B2 on the corrected set.
2. **GISTDA imagery + manual labels** (requires institutional access).
3. **OSM `roof:material=metal` tags** — query Thailand for tagged metal
   roofs, fetch tiles, use as weak labels for self-training.

See `.planning/` for the staged training plan if/when this is needed.

## Storage discipline

- `data/` directory is gitignored (see `.gitignore`).
- All inference happens via API → no local data needed for production.
- If you must cache imagery for development, use the MinIO bucket
  defined in `docker-compose.yml`, not local disk.
