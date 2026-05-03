"""Inference worker: dequeues jobs from Redis, runs two-stage pipeline, writes results.

This worker runs as a long-lived process (on RunPod persistent pod) that:
1. Polls the Redis quote_jobs queue
2. Fetches satellite tiles via Maxar/Nearmap API
3. Runs Stage 1 (roof footprint) + Stage 2 (corrosion segmentation)
4. Computes area, severity, confidence
5. Generates an itemized quote via the quote engine
6. Writes assessment + quote to Supabase
7. Updates job status in Redis + Supabase

Usage:
    python -m app.inference.worker [--poll-interval 5]

Can also be run via:
    docker compose run api python -m app.inference.worker
"""

import argparse
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch

from app.db import get_supabase
from app.inference.pipeline import CorrosionPipeline, classify_severity
from app.queue import (
    QUOTE_QUEUE,
    RELABEL_QUEUE,
    dequeue_job,
    enqueue_job,
    get_redis,
    set_job_status,
)
from app.quote_engine import compute_quote


# ── Model loading ───────────────────────────────────────────

def load_models(
    roof_model_uri: Optional[str] = None,
    corrosion_model_uri: Optional[str] = None,
    device: str = "auto",
):
    """Load the inference pipeline.

    Priority order (set via env):
    1. PIPELINE=fm (DEFAULT) + NVIDIA_API_KEY → Foundation-model pipeline (OSM + NIM VLM)
    2. PIPELINE=runpod  + RUNPOD_ENDPOINT_ID → trained models on RunPod Serverless
    3. PIPELINE=local   → load TorchScript/MLflow models locally
    4. Fallback → stub pipeline (returns zeros, no real inference)

    Returns CorrosionPipeline | RunPodCorrosionPipeline | FoundationModelPipeline
    """
    import os

    pipeline_mode = os.environ.get("PIPELINE", "fm").lower()

    # Option 1: Foundation-model pipeline (default, cheapest, zero training)
    if pipeline_mode == "fm":
        try:
            from app.inference.pipeline_fm import create_fm_pipeline
            pipeline = create_fm_pipeline()
            print(f"Using Foundation-Model pipeline: OSM + NIM "
                  f"({pipeline.corrosion_model_uri})")
            return pipeline
        except Exception as e:
            print(f"⚠️  Foundation-model pipeline init failed: {e}")
            print("   Falling back to RunPod / local pipeline...")

    # Option 2: RunPod Serverless (trained SegFormer on GPU)
    runpod_endpoint = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    runpod_key = os.environ.get("RUNPOD_API_KEY", "")
    if pipeline_mode in ("fm", "runpod") and runpod_endpoint and runpod_key:
        try:
            pipeline = CorrosionPipeline.from_runpod_serverless()
            print(f"Using RunPod Serverless endpoint: {runpod_endpoint}")
            return pipeline
        except Exception as e:
            print(f"⚠️  RunPod Serverless init failed: {e}")
            print("   Falling back to local pipeline...")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    roof_uri = roof_model_uri or "models:/roof_detector/production"
    corrosion_uri = corrosion_model_uri or "models:/corrosion_detector/production"

    try:
        pipeline = CorrosionPipeline(
            roof_model_uri=roof_uri,
            corrosion_model_uri=corrosion_uri,
            device=device,
        )
        print(f"Models loaded on {device}")
        return pipeline
    except Exception as e:
        print(f"⚠️  Failed to load models from MLflow: {e}")
        print("   Using stub pipeline (no real inference)")
        return CorrosionPipeline(
            roof_model_uri="stub",
            corrosion_model_uri="stub",
            device=device,
        )


# ── Tile fetching ───────────────────────────────────────────

async def fetch_tiles_for_job(job: dict) -> tuple[Optional[np.ndarray], float, tuple[float, float, float, float] | None]:
    """Fetch satellite tile(s) for a jobs location.

    Auto-selects the best available imagery source based on configured keys:
    Nearmap > Maxar > Mapbox > Esri (free, default).

    Returns:
        (tile_image, gsd, tile_bounds_latlng) where tile_bounds is
        (min_lat, min_lng, max_lat, max_lng) for the fetched tile, or None on failure.
    """
    from app.inference.tile_fetch import TileFetcher, TileRequest, ZOOM_TO_GSD_M

    fetcher = TileFetcher()
    # Use job-specified source if given, otherwise auto-select
    source = job.get("source") or fetcher.auto_select_source()
    lat = job.get("lat", 0.0)
    lng = job.get("lng", 0.0)

    # Pick zoom based on source (Maxar/Nearmap support zoom 20+, Esri/Mapbox up to 19-20)
    zoom = 20 if source in ("maxar", "nearmap") else 19

    try:
        request = TileRequest(lat=lat, lng=lng, zoom=zoom, source=source)
        tile_bytes = await fetcher.fetch_tile(request)

        from PIL import Image
        import io

        img = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
        tile_array = np.array(img)

        gsd = TileFetcher.gsd_for(source, zoom, lat=lat)

        # Compute the lat/lng bounds of this tile (used by FM pipeline to rasterize OSM polygons)
        tile_bounds = _tile_bounds_latlng(lat, lng, zoom)

        print(f"Fetched tile: source={source} zoom={zoom} gsd={gsd:.2f}m {tile_array.shape}")
        return tile_array, gsd, tile_bounds
    except Exception as e:
        print(f"⚠️  Failed to fetch tiles ({source}): {e}")
        return None, 0.0, None


def _tile_bounds_latlng(lat: float, lng: float, zoom: int) -> tuple[float, float, float, float]:
    """Compute the (min_lat, min_lng, max_lat, max_lng) bounds of the tile at (lat, lng, zoom)."""
    import math
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)

    def _tile_to_latlng(x: int, y: int, n: int) -> tuple[float, float]:
        lng = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad), lng

    nw_lat, nw_lng = _tile_to_latlng(x, y, n)
    se_lat, se_lng = _tile_to_latlng(x + 1, y + 1, n)
    return (se_lat, nw_lng, nw_lat, se_lng)


# ── Geocoding ────────────────────────────────────────────────

async def geocode_address(address: str) -> tuple[float, float]:
    """Geocode an address to lat/lng using Nominatim (free), biased to active region.

    The active region (default: Thailand) is passed to Nominatim via
    `countrycodes` and `viewbox` to prefer in-region matches. Out-of-region
    results raise a RegionError — caller can surface this to the UI.
    """
    import httpx
    from app.region import get_active_region, is_in_region

    region = get_active_region()
    params: dict = {"q": address, "format": "json", "limit": 3}

    # Bias to active region
    if region.country_code:
        params["countrycodes"] = region.country_code
    if region.bbox:
        min_lat, min_lng, max_lat, max_lng = region.bbox
        # Nominatim viewbox format: minlng,minlat,maxlng,maxlat
        params["viewbox"] = f"{min_lng},{min_lat},{max_lng},{max_lat}"
        params["bounded"] = 1
    if region.osm_locale:
        params["accept-language"] = f"{region.osm_locale},en"

    url = "https://nominatim.openstreetmap.org/search"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, params=params,
            headers={"User-Agent": "roof-corrosion-ai/0.1"},
        )
        data = response.json()

    for match in data or []:
        lat = float(match["lat"])
        lng = float(match["lon"])
        if is_in_region(lat, lng, region):
            return lat, lng

    raise ValueError(
        f"No in-region ({region.code}) geocoding result for: {address}. "
        f"Set REGION env var if you meant a different country."
    )


# ── Main worker loop ─────────────────────────────────────────

def process_job(job: dict, pipeline) -> dict:
    """Process a single inference job synchronously.

    Steps:
    1. Geocode address if needed
    2. Fetch satellite tile
    3. Run two-stage inference
    4. Compute quote
    5. Write results to Supabase
    """
    import asyncio

    job_id = job["job_id"]
    start_time = time.time()

    # Update status
    set_job_status(job_id, {"job_id": job_id, "status": "processing"})
    update_job_in_supabase(job_id, {"status": "processing", "started_at": datetime.utcnow().isoformat()})

    try:
        # 1. Geocode if address provided but no coords
        if job.get("address") and not job.get("lat"):
            lat, lng = asyncio.run(geocode_address(job["address"]))
            job["lat"] = lat
            job["lng"] = lng
            update_job_in_supabase(job_id, {"latitude": lat, "longitude": lng})

        # 2. Fetch tile
        tile_array, gsd, tile_bounds = asyncio.run(fetch_tiles_for_job(job))
        if tile_array is None:
            raise RuntimeError("Failed to fetch satellite tile")

        update_job_in_supabase(job_id, {"gsd_m": gsd})

        # Pass tile bounds through to the FM pipeline for OSM polygon rasterization
        if tile_bounds is not None:
            job["tile_bounds"] = tile_bounds

        # 3. Run inference — dispatch based on pipeline type
        from app.inference.pipeline_fm import FoundationModelPipeline
        from app.inference.pipeline import RunPodCorrosionPipeline

        if isinstance(pipeline, FoundationModelPipeline):
            # FM pipeline is async — takes address + coords for OSM + VLM context
            result = asyncio.run(pipeline.analyze(
                tile_image=tile_array,
                lat=job.get("lat"),
                lng=job.get("lng"),
                gsd=gsd,
                address=job.get("address", ""),
                tile_bounds=job.get("tile_bounds"),
            ))
        elif isinstance(pipeline, RunPodCorrosionPipeline):
            result = asyncio.run(pipeline.analyze_async(tile_array, gsd=gsd))
        else:
            # Legacy sync pipeline
            result = pipeline.analyze(tile_array, gsd=gsd)

        # 4. Compute quote
        quote_result = compute_quote(
            roof_area_m2=result.roof_area_m2,
            corroded_area_m2=result.corroded_area_m2,
            corrosion_percent=result.corrosion_percent,
            severity=result.severity,
            confidence=result.confidence,
            material=job.get("material", "corrugated_metal"),
            region=job.get("region", "default"),
        )

        # 5. Write results to Supabase
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Write assessment
        assessment_id = write_assessment_to_supabase(job_id, result, gsd)

        # Write quote
        write_quote_to_supabase(job_id, assessment_id, quote_result)

        # Update job as completed
        update_job_in_supabase(job_id, {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time_ms,
            "roof_model_version": pipeline.roof_model_uri or "stub",
            "corrosion_model_version": pipeline.corrosion_model_uri or "stub",
        })

        final_status = {
            "job_id": job_id,
            "status": "completed",
            "assessment": {
                "roof_area_m2": result.roof_area_m2,
                "corroded_area_m2": result.corroded_area_m2,
                "corrosion_percent": result.corrosion_percent,
                "severity": result.severity,
                "confidence": result.confidence,
            },
            "quote": {
                "total_amount": quote_result.total_amount,
                "currency": quote_result.currency,
                "line_items": quote_result.line_items,
                "requires_human_review": quote_result.requires_human_review,
            },
            "processing_time_ms": processing_time_ms,
        }

        set_job_status(job_id, final_status)
        return final_status

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

        update_job_in_supabase(job_id, {
            "status": "failed",
            "error_message": error_msg,
            "completed_at": datetime.utcnow().isoformat(),
        })
        set_job_status(job_id, {"job_id": job_id, "status": "failed", "error": error_msg})

        return {"job_id": job_id, "status": "failed", "error": error_msg}


def update_job_in_supabase(job_id: str, updates: dict) -> None:
    """Update job record in Supabase."""
    try:
        supabase = get_supabase()
        supabase.table("jobs").update(updates).eq("id", job_id).execute()
    except Exception as e:
        print(f"⚠️  Failed to update Supabase job {job_id}: {e}")


def write_assessment_to_supabase(job_id: str, result, gsd: float) -> str:
    """Write assessment to Supabase. Returns assessment ID."""
    import uuid

    assessment_id = str(uuid.uuid4())
    try:
        supabase = get_supabase()
        supabase.table("assessments").insert({
            "id": assessment_id,
            "job_id": job_id,
            "roof_area_m2": result.roof_area_m2,
            "corroded_area_m2": result.corroded_area_m2,
            "corrosion_percent": result.corrosion_percent,
            "severity": result.severity,
            "confidence": result.confidence,
        }).execute()
    except Exception as e:
        print(f"⚠️  Failed to write assessment: {e}")

    return assessment_id


def write_quote_to_supabase(job_id: str, assessment_id: str, quote_result) -> None:
    """Write quote to Supabase."""
    import uuid

    quote_id = str(uuid.uuid4())
    try:
        supabase = get_supabase()
        supabase.table("quotes").insert({
            "id": quote_id,
            "job_id": job_id,
            "assessment_id": assessment_id,
            "currency": quote_result.currency,
            "total_amount": quote_result.total_amount,
            "line_items": quote_result.line_items,
            "requires_human_review": quote_result.requires_human_review,
            "valid_until": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        }).execute()
    except Exception as e:
        print(f"⚠️  Failed to write quote: {e}")


def run_worker(poll_interval: int = 5, device: str = "auto"):
    """Main worker loop. Polls Redis queue and processes jobs."""
    print("=" * 60)
    print("Roof Corrosion AI — Inference Worker")
    print("=" * 60)

    # Load models
    pipeline = load_models(device=device)

    print(f"Polling {QUOTE_QUEUE} every {poll_interval}s...")
    print("Press Ctrl+C to stop.\n")

    jobs_processed = 0
    while True:
        try:
            job = dequeue_job(QUOTE_QUEUE, timeout=poll_interval)
            if job:
                print(f"\n▶ Processing job {job.get('job_id', 'unknown')}")
                result = process_job(job, pipeline)
                jobs_processed += 1

                status = result.get("status", "unknown")
                if status == "completed":
                    assessment = result.get("assessment", {})
                    print(f"  ✓ Completed: severity={assessment.get('severity')}, "
                          f"corrosion={assessment.get('corrosion_percent', 0):.1f}%, "
                          f"area={assessment.get('roof_area_m2', 0):.0f}m²")
                else:
                    print(f"  ✗ Failed: {result.get('error', 'unknown')}")

                print(f"  Jobs processed this session: {jobs_processed}")
            else:
                # No job available, brief idle
                pass

        except KeyboardInterrupt:
            print(f"\n\nWorker stopped. {jobs_processed} jobs processed.")
            break
        except Exception as e:
            print(f"⚠️  Worker error: {e}")
            traceback.print_exc()
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roof Corrosion AI Inference Worker")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between queue polls")
    parser.add_argument("--device", type=str, default="auto", help="torch device (auto/cuda/cpu)")
    args = parser.parse_args()
    run_worker(poll_interval=args.poll_interval, device=args.device)
