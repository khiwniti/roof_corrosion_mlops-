"""Quote submission and retrieval endpoints — wired to Supabase + Redis."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import get_supabase
from app.queue import enqueue_job, get_job_status, QUOTE_QUEUE

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
    except Exception as e:
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
