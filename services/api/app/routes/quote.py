"""Quote submission and retrieval endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class QuoteRequest(BaseModel):
    address: str | None = Field(None, description="Street address to geocode")
    lat: float | None = Field(None, description="Latitude (if address not provided)")
    lng: float | None = Field(None, description="Longitude (if address not provided)")
    polygon_geojson: dict | None = Field(None, description="Custom AOI polygon")


class QuoteResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/", response_model=QuoteResponse)
async def submit_quote_request(req: QuoteRequest) -> QuoteResponse:
    """Submit a roof corrosion analysis job."""
    # TODO: geocode address → fetch tiles → enqueue inference job
    return QuoteResponse(
        job_id="stub-job-id",
        status="queued",
        message="Job submitted. Poll /quote/{job_id} for results.",
    )


@router.get("/{job_id}")
async def get_quote_status(job_id: str) -> dict[str, str]:
    """Get the status and results of a quote job."""
    # TODO: look up job in Redis/Postgres, return result if done
    return {"job_id": job_id, "status": "processing"}
