"""GISTDA THEOS-2 commercial imagery API client.

Architecture Decision: ADR-001
- Source: GISTDA THEOS-2 (Thai-registered pricing by request)
- GSD: 0.5 m pan / ~2 m MS
- Coverage: Thailand prioritized

Note: THEOS-2 API is not publicly documented. This module is a stub
based on typical EO catalog APIs (STAC-like). Update endpoints when
commercial contract is signed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from shapely.geometry import shape

logger = logging.getLogger("ingestion.theos2")

# Placeholder endpoints — update after GISTDA contract
THEOS2_BASE_URL = os.environ.get("THEOS2_API_URL", "https://catalog.gistda.or.th/api/v1")
THEOS2_AUTH_URL = f"{THEOS2_BASE_URL}/auth/token"


def _get_access_token() -> str | None:
    """Authenticate with GISTDA THEOS-2 API."""
    client_id = os.environ.get("THEOS2_CLIENT_ID")
    client_secret = os.environ.get("THEOS2_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.warning("THEOS2_CLIENT_ID/SECRET not set")
        return None
    try:
        import requests
        resp = requests.post(
            THEOS2_AUTH_URL,
            json={"client_id": client_id, "client_secret": client_secret},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        logger.warning("THEOS-2 auth failed: %s", e)
        return None


def search_catalog(
    polygon_geojson: dict[str, Any],
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    max_cloud: float = 20.0,
    max_items: int = 20,
) -> list[dict[str, Any]]:
    """Search THEOS-2 catalog for scenes overlapping a polygon.

    Stub: returns empty list. Implement when API contract is available.
    """
    token = _get_access_token()
    if not token:
        logger.warning("Cannot search THEOS-2 without auth token")
        return []

    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    aoi = shape(polygon_geojson)
    bbox = aoi.bounds

    logger.info(
        "THEOS-2 search stub: bbox=%s, cloud<%s, %s–%s",
        bbox, max_cloud, start_date, end_date,
    )
    return []


def estimate_cost(
    polygon_geojson: dict[str, Any],
) -> dict[str, Any]:
    """Estimate THEOS-2 imagery cost.

    Stub: returns placeholder. Real pricing by request from GISTDA.
    """
    aoi = shape(polygon_geojson)
    area_km2 = aoi.area * 111.32 * 111.32

    return {
        "area_km2": round(area_km2, 3),
        "constellation": "THEOS-2",
        "rate_thb_per_km2": None,
        "estimated_cost_thb": None,
        "note": "Pricing available by request from GISTDA (Thai-registered entity preferred).",
        "contact": "https://www.gistda.or.th/contact",
    }
