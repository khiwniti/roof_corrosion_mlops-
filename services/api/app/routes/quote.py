"""Quote submission and retrieval endpoints -- wired to Supabase + Redis."""

import logging
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import get_supabase
from app.queue import enqueue_job, get_job_status, QUOTE_QUEUE
from app.region import get_active_region

logger = logging.getLogger(__name__)
router = APIRouter()


class QuoteRequest(BaseModel):
    address: str | None = Field(None, description="Street address to geocode")
    lat: float | None = Field(None, description="Latitude")
    lng: float | None = Field(None, description="Longitude")
    polygon_geojson: dict | None = Field(None, description="Custom AOI polygon")
    customer_id: str | None = Field(None, description="Authenticated customer ID")
    material: str = Field("corrugated_metal", description="Roof material type")
    region: str = Field("default", description="Pricing region")


class QuoteResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/", response_model=QuoteResponse)
async def submit_quote_request(req: QuoteRequest) -> QuoteResponse:
    """Submit a roof corrosion analysis job.

    Creates a job in Supabase, enqueues for inference processing.
    """
    if not req.address and (req.lat is None or req.lng is None):
        raise HTTPException(status_code=400, detail="Provide either address or lat/lng coordinates")

    job_id = str(uuid.uuid4())

    # Create job in Supabase
    try:
        supabase = get_supabase()
        supabase.table("jobs").insert({
            "id": job_id,
            "customer_id": req.customer_id or "00000000-0000-0000-0000-000000000000",
            "status": "queued",
            "source": "maxar",
            "address": req.address,
            "latitude": req.lat,
            "longitude": req.lng,
            "aoi_geojson": req.polygon_geojson,
        }).execute()
    except Exception:
        # Fallback: just enqueue without DB (for local dev)
        pass

    # Enqueue for inference
    enqueue_job(QUOTE_QUEUE, {
        "job_id": job_id,
        "address": req.address,
        "lat": req.lat,
        "lng": req.lng,
        "aoi_geojson": req.polygon_geojson,
        "material": req.material,
        "region": req.region,
    })

    return QuoteResponse(
        job_id=job_id,
        status="queued",
        message="Job submitted. Poll /quote/{job_id} for results.",
    )


@router.get("/{job_id}")
async def get_quote_status(job_id: str) -> dict:
    """Get the status and results of a quote job."""
    # Check Redis for real-time status
    redis_status = get_job_status(job_id)
    if redis_status:
        return redis_status

    # Check Supabase for completed jobs
    try:
        supabase = get_supabase()
        job = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
        if job.data:
            result = {"job_id": job_id, "status": job.data["status"]}
            # If completed, include assessment + quote
            if job.data["status"] == "completed":
                assessment = supabase.table("assessments").select("*").eq("job_id", job_id).single().execute()
                if assessment.data:
                    result["assessment"] = assessment.data
                quote = supabase.table("quotes").select("*").eq("job_id", job_id).single().execute()
                if quote.data:
                    result["quote"] = quote.data
            return result
    except Exception:
        pass

    return {"job_id": job_id, "status": "unknown"}


# ========================================================================
# Synchronous endpoint -- runs the FM pipeline inline (no Redis/worker)
# ========================================================================

class SyncQuoteResponse(BaseModel):
    """Result of a synchronous /quote/sync request."""
    job_id: str
    status: str
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    gsd_m: float
    tile_source: str
    assessment: dict
    quote: dict
    processing_time_ms: int
    model_versions: dict


async def _proxy_to_serverless(
    req: QuoteRequest, endpoint_id: str, api_key: str
) -> SyncQuoteResponse:
    """Proxy the inference request to the RunPod Serverless endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "lat": req.lat,
            "lng": req.lng,
            "address": req.address,
            "material": req.material,
            "region": req.region if req.region != "default" else None,
        }
    }

    t0 = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod serverless returned {resp.status_code}: {resp.text[:500]}",
        )

    data = resp.json()

    # Handle errors from the worker
    if data.get("error"):
        raise HTTPException(status_code=502, detail=f"Inference error: {data['error']}")

    output = data.get("output", {})
    processing_time_ms = int((time.time() - t0) * 1000)

    # Map serverless output to SyncQuoteResponse
    assessment = output.get("assessment", {})
    quote = output.get("quote", {})
    models = output.get("model_versions", {})

    return SyncQuoteResponse(
        job_id=data.get("id", str(uuid.uuid4())),
        status=output.get("status", "completed"),
        address=req.address,
        lat=req.lat,
        lng=req.lng,
        gsd_m=output.get("gsd_m", 0.3),
        tile_source=models.get("roof", "serverless"),
        assessment=assessment,
        quote=quote,
        processing_time_ms=processing_time_ms,
        model_versions=models,
    )


async def _run_local_pipeline(req: QuoteRequest) -> SyncQuoteResponse:
    """Run the FM pipeline locally (requires full ML stack)."""
    # Lazy imports -- only loaded when this endpoint is hit
    from app.inference.pipeline_fm import create_fm_pipeline
    from app.inference.worker import fetch_tiles_for_job, geocode_address
    from app.quote_engine import compute_quote

    job_id = str(uuid.uuid4())
    region = get_active_region()
    t0 = time.time()

    # 1. Geocode if needed
    lat, lng = req.lat, req.lng
    if (lat is None or lng is None) and req.address:
        try:
            lat, lng = await geocode_address(req.address)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Geocoding failed: {e}") from e

    # 2. Fetch satellite tile
    job_for_fetch = {"lat": lat, "lng": lng, "source": None}
    tile_array, gsd, tile_bounds = await fetch_tiles_for_job(job_for_fetch)
    if tile_array is None:
        raise HTTPException(status_code=502, detail="Failed to fetch satellite tile")

    # 3. Run FM pipeline (OSM + NIM VLM)
    try:
        pipeline = create_fm_pipeline()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"FM pipeline init failed (set NVIDIA_API_KEY): {e}",
        ) from e

    try:
        result = await pipeline.analyze(
            tile_image=tile_array, lat=lat, lng=lng, gsd=gsd,
            address=req.address or "", tile_bounds=tile_bounds,
        )
    except Exception as e:
        logger.exception("FM pipeline failed")
        raise HTTPException(status_code=502, detail=f"Inference failed: {e}") from e

    # 4. Compute quote
    quote_result = compute_quote(
        roof_area_m2=result.roof_area_m2,
        corroded_area_m2=result.corroded_area_m2,
        corrosion_percent=result.corrosion_percent,
        severity=result.severity,
        confidence=result.confidence,
        material=req.material,
        region=req.region if req.region != "default" else region.code,
    )

    processing_time_ms = int((time.time() - t0) * 1000)

    return SyncQuoteResponse(
        job_id=job_id,
        status="completed",
        address=req.address,
        lat=lat,
        lng=lng,
        gsd_m=gsd,
        tile_source=str(getattr(pipeline, "_last_tile_source", "auto")),
        assessment={
            "roof_area_m2": round(result.roof_area_m2, 2),
            "corroded_area_m2": round(result.corroded_area_m2, 2),
            "corrosion_percent": round(result.corrosion_percent, 1),
            "severity": result.severity,
            "confidence": round(result.confidence, 3),
        },
        quote={
            "currency": quote_result.currency,
            "total_amount": quote_result.total_amount,
            "line_items": quote_result.line_items,
            "requires_human_review": quote_result.requires_human_review,
            "review_reason": quote_result.review_reason,
        },
        processing_time_ms=processing_time_ms,
        model_versions={
            "roof": pipeline.roof_model_uri,
            "corrosion": pipeline.corrosion_model_uri,
        },
    )


@router.post("/sync", response_model=SyncQuoteResponse)
async def submit_quote_sync(req: QuoteRequest) -> SyncQuoteResponse:
    """Run the foundation-model pipeline and return the result immediately.

    Two modes:
    1. **Serverless proxy** (default): Forwards to the RunPod Serverless
       inference endpoint when RUNPOD_ENDPOINT_ID is set.
    2. **Local pipeline**: Runs the FM pipeline inline when the full
       ML stack (torch, rasterio) is available and no endpoint ID is set.

    Use this for demos and simple integrations. For high-throughput
    production, use POST /quote (async with Redis worker).
    """
    if not req.address and (req.lat is None or req.lng is None):
        raise HTTPException(status_code=400, detail="Provide either address or lat/lng coordinates")

    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")

    # Serverless proxy mode (default for production)
    if endpoint_id and api_key:
        return await _proxy_to_serverless(req, endpoint_id, api_key)

    # Local pipeline mode (requires full ML stack)
    return await _run_local_pipeline(req)
