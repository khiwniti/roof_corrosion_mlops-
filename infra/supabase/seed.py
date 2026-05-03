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

    # Sample job — Thai industrial warehouse in Samut Prakan
    job_id = str(uuid4())
    print(f"Creating sample job: {job_id}")
    supabase.table("jobs").insert({
        "id": job_id,
        "customer_id": customer_id,
        "status": "completed",
        "source": "maxar",
        "address": "88 Bangna-Trad Rd, Bang Sao Thong, Samut Prakan 10540, Thailand",
        "latitude": 13.5924,
        "longitude": 100.7866,
        "gsd_m": 0.3,
        "tile_capture_date": "2024-06-15",
        "submitted_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "started_at": (datetime.utcnow() - timedelta(hours=2, minutes=1)).isoformat(),
        "completed_at": (datetime.utcnow() - timedelta(hours=1, minutes=55)).isoformat(),
        "processing_time_ms": 45000,
        "roof_model_version": "osm+nim/llama-3.2-90b-vision",
        "corrosion_model_version": "nvidia-nim/llama-3.2-90b-vision-instruct",
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

    # Quote — Thailand pricing in THB
    quote_id = str(uuid4())
    print(f"Creating quote: {quote_id}")
    # Pricing: THB 450/m² repair, THB 280/m² coating, THB 850/m² replacement
    repair_total = round(85.5 * 850.00, 2)       # corroded section replacement
    coating_area = round(320.0 - 85.5, 1)
    coating_total = round(coating_area * 280.00, 2)
    waste_qty = round(85.5 * 0.10, 1)
    waste_total = round(waste_qty * 850.00, 2)
    inspection_total = 2500.00
    grand_total = inspection_total + repair_total + coating_total + waste_total
    supabase.table("quotes").insert({
        "id": quote_id,
        "job_id": job_id,
        "assessment_id": assessment_id,
        "currency": "THB",
        "total_amount": round(grand_total, 2),
        "line_items": [
            {"description": "Satellite roof inspection & corrosion analysis", "quantity": 1, "unit": "each", "unit_price": 2500.00, "total": 2500.00},
            {"description": "Corroded section replacement (moderate)", "quantity": 85.5, "unit": "m²", "unit_price": 850.00, "total": repair_total},
            {"description": "Protective anti-rust coating (remaining roof)", "quantity": coating_area, "unit": "m²", "unit_price": 280.00, "total": coating_total},
            {"description": "Waste allowance (10%)", "quantity": waste_qty, "unit": "m²", "unit_price": 850.00, "total": waste_total},
        ],
        "requires_human_review": False,
        "valid_until": (datetime.utcnow() + timedelta(days=30)).isoformat(),
    }).execute()

    print(f"\nSeed data created:")
    print(f"  Customer: test@roofcorrosion.ai")
    print(f"  Job:      {job_id}")
    print(f"  Address:  Bangna-Trad Rd, Samut Prakan, Thailand")
    print(f"  Quote:    THB {grand_total:,.2f} (moderate corrosion, 26.7%)")


if __name__ == "__main__":
    seed()
