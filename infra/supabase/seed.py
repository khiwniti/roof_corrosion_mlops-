"""Supabase seed script for local development.

Creates a test customer, a sample job, assessment, and quote
so the frontend can render real data without running inference.

Usage:
    python infra/supabase/seed.py

Requires: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars.
"""

import os
import sys
from datetime import datetime, timedelta
from uuid import uuid4

try:
    from supabase import create_client
except ImportError:
    print("Install supabase-py: pip install supabase")
    sys.exit(1)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def seed():
    # Test customer
    customer_id = str(uuid4())
    print(f"Creating test customer: {customer_id}")
    supabase.table("customers").insert({
        "id": customer_id,
        "email": "test@roofcorrosion.ai",
        "full_name": "Test Customer",
        "company_name": "Test Industrial Corp",
    }).execute()

    # Sample job
    job_id = str(uuid4())
    print(f"Creating sample job: {job_id}")
    supabase.table("jobs").insert({
        "id": job_id,
        "customer_id": customer_id,
        "status": "completed",
        "source": "maxar",
        "address": "Jl. Industri Raya No. 42, Jakarta Industrial Park, Cakung, Jakarta Timur",
        "latitude": -6.2088,
        "longitude": 106.8456,
        "gsd_m": 0.3,
        "tile_capture_date": "2024-06-15",
        "submitted_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "started_at": (datetime.utcnow() - timedelta(hours=2, minutes=1)).isoformat(),
        "completed_at": (datetime.utcnow() - timedelta(hours=1, minutes=55)).isoformat(),
        "processing_time_ms": 45000,
        "roof_model_version": "roof_detector/v0.1.0-dev",
        "corrosion_model_version": "corrosion_detector/v0.1.0-dev",
    }).execute()

    # Assessment
    assessment_id = str(uuid4())
    print(f"Creating assessment: {assessment_id}")
    supabase.table("assessments").insert({
        "id": assessment_id,
        "job_id": job_id,
        "roof_area_m2": 320.0,
        "corroded_area_m2": 85.5,
        "corrosion_percent": 26.7,
        "severity": "moderate",
        "confidence": 0.78,
    }).execute()

    # Quote
    quote_id = str(uuid4())
    print(f"Creating quote: {quote_id}")
    supabase.table("quotes").insert({
        "id": quote_id,
        "job_id": job_id,
        "assessment_id": assessment_id,
        "currency": "USD",
        "total_amount": 7155.00,
        "line_items": [
            {"description": "Satellite roof inspection & corrosion analysis", "quantity": 1, "unit": "each", "unit_price": 150.00, "total": 150.00},
            {"description": "Corroded section replacement (moderate)", "quantity": 85.5, "unit": "m²", "unit_price": 45.00, "total": 3847.50},
            {"description": "Protective coating (remaining roof)", "quantity": 234.5, "unit": "m²", "unit_price": 15.00, "total": 3517.50},
            {"description": "Waste allowance (10%)", "quantity": 8.6, "unit": "m²", "unit_price": 45.00, "total": 387.00},
        ],
        "requires_human_review": False,
        "valid_until": (datetime.utcnow() + timedelta(days=30)).isoformat(),
    }).execute()

    print(f"\nSeed data created:")
    print(f"  Customer: test@roofcorrosion.ai")
    print(f"  Job:      {job_id}")
    print(f"  Quote:    USD 7,155.00 (moderate corrosion, 26.7%)")


if __name__ == "__main__":
    seed()
