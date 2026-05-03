"""Database connection and models via Supabase."""

import os
from typing import Optional

from pydantic import BaseModel, Field


# ── Pydantic models (mirrors Supabase schema) ──────────────

class Customer(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    company_name: Optional[str] = None


class JobCreate(BaseModel):
    customer_id: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    aoi_geojson: Optional[dict] = None
    source: str = "maxar"
    tile_zoom: int = 20


class Job(JobCreate):
    id: str
    status: str = "queued"
    tile_capture_date: Optional[str] = None
    gsd_m: Optional[float] = None
    submitted_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None
    roof_model_version: Optional[str] = None
    corrosion_model_version: Optional[str] = None
    mlflow_run_id: Optional[str] = None


class Assessment(BaseModel):
    id: str
    job_id: str
    roof_area_m2: float
    corroded_area_m2: float
    corrosion_percent: float
    severity: str = "none"
    confidence: float = 0.0
    roof_mask_s3_key: Optional[str] = None
    corrosion_mask_s3_key: Optional[str] = None
    overlay_image_s3_key: Optional[str] = None


class LineItem(BaseModel):
    description: str
    quantity: float
    unit: str
    unit_price: float
    total: float


class Quote(BaseModel):
    id: str
    job_id: str
    assessment_id: str
    currency: str = "USD"
    total_amount: float
    line_items: list[LineItem]
    requires_human_review: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    approved: Optional[bool] = None
    pdf_s3_key: Optional[str] = None
    valid_until: Optional[str] = None


class FeedbackCreate(BaseModel):
    job_id: str
    customer_id: str
    correct: bool
    notes: Optional[str] = None
    roof_boundary_wrong: bool = False
    corrosion_area_wrong: bool = False
    severity_wrong: bool = False


class PriceBookEntry(BaseModel):
    material: str
    service_type: str
    region: str = "default"
    price_per_m2: float
    currency: str = "USD"


# ── Supabase client ─────────────────────────────────────────

from supabase import create_client, Client

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get or create Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("Supabase credentials not configured. Set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client
