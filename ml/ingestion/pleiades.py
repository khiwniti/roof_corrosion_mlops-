"""Airbus OneAtlas Pléiades archive search and download.

Architecture Decision: ADR-001
- Source: Airbus OneAtlas API (Pléiades archive / Neo archive / Neo tasking)
- Pricing: €3.80/km² archive, €18/km² Neo archive, €25–40/km² Neo tasking
- Aggregation: neighborhood-block level (1 km² scene → 50–200 quotes)

API docs: https://api.oneatlas.airbus.com/
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import requests
from shapely.geometry import Polygon, shape

logger = logging.getLogger("ingestion.pleiades")

ONEATLAS_BASE = "https://api.oneatlas.airbus.com"
ONEATLAS_AUTH_URL = f"{ONEATLAS_BASE}/api/v1/login"
ONEATLAS_SEARCH_URL = f"{ONEATLAS_BASE}/api/v1/opensearch"


def _get_access_token() -> str | None:
    """Authenticate with OneAtlas and return access token."""
    api_key = os.environ.get("ONEATLAS_API_KEY")
    if not api_key:
        logger.warning("ONEATLAS_API_KEY not set")
        return None
    try:
        resp = requests.post(
            ONEATLAS_AUTH_URL,
            json={"apiKey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        logger.warning("OneAtlas auth failed: %s", e)
        return None


def search_archive(
    polygon_geojson: dict[str, Any],
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    max_cloud: float = 15.0,
    constellation: str = "PHR",  # PHR = Pléiades, SP6 = SPOT-6/7
    max_items: int = 20,
) -> list[dict[str, Any]]:
    """Search OneAtlas archive for scenes overlapping a polygon.

    Parameters
    ----------
    polygon_geojson : GeoJSON Polygon dict
    start_date, end_date : ISO date strings or datetime. Defaults to last 90 days.
    max_cloud : max cloud cover percentage
    constellation : "PHR" | "SP6" | "PNEO"
    max_items : max results to return

    Returns
    -------
    list of scene metadata dicts
    """
    token = _get_access_token()
    if not token:
        logger.warning("Cannot search OneAtlas without auth token")
        return []

    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    aoi = shape(polygon_geojson)
    bbox = aoi.bounds  # (minx, miny, maxx, maxy)

    # OneAtlas uses bbox query + cloud cover filter
    # Full WKT polygon can also be passed as "geometry" parameter
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "constellation": constellation,
        "cloudCover": f"[0,{max_cloud}]",
        "acquisitionDate": f"[{start_date.strftime('%Y-%m-%d')}T00:00:00Z,{end_date.strftime('%Y-%m-%d')}T23:59:59Z]",
        "count": max_items,
    }

    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(ONEATLAS_SEARCH_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        logger.info("OneAtlas search: %s scenes for AOI", len(features))
        return features
    except Exception as e:
        logger.warning("OneAtlas search failed: %s", e)
        return []


def estimate_cost(
    polygon_geojson: dict[str, Any],
    constellation: str = "PHR",
) -> dict[str, Any]:
    """Estimate imagery cost for an AOI.

    Returns cost in EUR and THB (approximate).
    """
    aoi = shape(polygon_geojson)
    area_km2 = aoi.area * 111.32 * 111.32  # rough conversion deg² → km² at equator

    rates = {
        "PHR": 3.80,
        "SP6": 2.50,
        "PNEO": 18.0,
    }
    rate = rates.get(constellation, 3.80)
    eur = area_km2 * rate
    thb = eur * 38.5  # approximate EUR→THB

    return {
        "area_km2": round(area_km2, 3),
        "constellation": constellation,
        "rate_eur_per_km2": rate,
        "estimated_cost_eur": round(eur, 2),
        "estimated_cost_thb": round(thb, 2),
        "aggregation_note": "Purchase at neighborhood-block level to amortize across 50–200 quotes",
    }


def download_quicklook(
    scene_id: str,
    output_path: str,
) -> bool:
    """Download a quicklook (low-res preview) for a scene.

    Stub: logs the call. Production uses OneAtlas download endpoint.
    """
    logger.info("Downloading quicklook for %s → %s (stub)", scene_id, output_path)
    return True


def order_archive_scene(
    scene_id: str,
    polygon_geojson: dict[str, Any],
    product_type: str = " Panchromatic 50cm",
) -> dict[str, Any]:
    """Place an order for an archive scene.

    Stub: logs the call. Production uses OneAtlas ordering API.
    """
    logger.info("Ordering archive scene %s (stub)", scene_id)
    return {
        "order_id": f"stub-order-{scene_id}",
        "status": "pending",
        "estimated_delivery": "near-instant",
        "product_type": product_type,
    }
