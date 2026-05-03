"""Building-footprint client — no training needed, free coverage.

Primary source: OpenStreetMap Overpass API (free, global, realtime)
  → queries building polygons around a lat/lng

Secondary source: Microsoft Global ML Building Footprints
  → 1.4B building polygons worldwide, released as open data on GitHub + Azure
  → static tiles by country, require download

Tertiary fallback: Google Open Buildings (Africa, SE Asia, South America)

This client prefers OSM since it's realtime and doesn't require bulk downloads.
For addresses where OSM is sparse, we fall back to a bbox from the tile itself.

Usage:
    client = BuildingFootprintClient()
    footprint = await client.get_footprint(lat=40.7128, lng=-74.0060, radius_m=50)
    # → {"polygon_ll": [[lat, lng], ...], "bbox": [...], "source": "osm"}
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"


# ═══════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════

def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    """Convert meters to (lat_deg, lng_deg) at a given latitude."""
    # 1 degree of latitude ≈ 111,320 meters (~constant)
    lat_deg = meters / 111_320.0
    # 1 degree of longitude = 111,320 * cos(lat) meters
    lng_deg = meters / (111_320.0 * math.cos(math.radians(lat)))
    return lat_deg, lng_deg


def polygon_area_m2(polygon_ll: list[list[float]]) -> float:
    """Approximate area of a polygon given in [lat, lng] coordinates.

    Uses the planar approximation (shoelace) after converting to local
    meters. Good enough for building-sized polygons.
    """
    if len(polygon_ll) < 3:
        return 0.0

    lat0 = polygon_ll[0][0]
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lng = 111_320.0 * math.cos(math.radians(lat0))

    # Convert to local (x, y) in meters
    xy = [
        ((p[1] - polygon_ll[0][1]) * meters_per_deg_lng,
         (p[0] - polygon_ll[0][0]) * meters_per_deg_lat)
        for p in polygon_ll
    ]

    # Shoelace
    area = 0.0
    n = len(xy)
    for i in range(n):
        j = (i + 1) % n
        area += xy[i][0] * xy[j][1]
        area -= xy[j][0] * xy[i][1]
    return abs(area) / 2.0


# ═══════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════

class BuildingFootprintClient:
    """Fetch building polygon(s) around a lat/lng from OSM (or fallback)."""

    def __init__(
        self,
        overpass_url: str = OVERPASS_ENDPOINT,
        timeout: float = 30.0,
        user_agent: str = "roof-corrosion-ai/0.1",
    ):
        self.overpass_url = overpass_url
        self.timeout = timeout
        self.user_agent = user_agent

    async def get_footprint(
        self,
        lat: float,
        lng: float,
        radius_m: float = 50.0,
    ) -> Optional[dict]:
        """Get the building polygon that contains (or is nearest to) lat/lng.

        Args:
            lat, lng: WGS84 coordinates of the address
            radius_m: search radius in meters

        Returns:
            {
                "polygon_ll": [[lat, lng], ...],   # outer ring
                "bbox": [min_lat, min_lng, max_lat, max_lng],
                "area_m2": float,
                "source": "osm" | "bbox",
                "osm_id": str | None,
                "tags": dict,
            }
            or None if no building found.
        """
        try:
            buildings = await self._query_osm(lat, lng, radius_m)
        except Exception as e:
            logger.warning(f"OSM query failed: {e} — falling back to bbox")
            return self._make_bbox_footprint(lat, lng, radius_m)

        if not buildings:
            # No OSM building tagged here — return a bbox around the point
            logger.info(f"No OSM building at ({lat}, {lng}); using bbox fallback")
            return self._make_bbox_footprint(lat, lng, radius_m)

        # Pick the building closest to the query point (by centroid)
        best = min(buildings, key=lambda b: self._distance_to_centroid(lat, lng, b["polygon_ll"]))
        return best

    async def _query_osm(self, lat: float, lng: float, radius_m: float) -> list[dict]:
        """Query Overpass for all buildings within radius."""
        # Overpass QL: ways tagged building=* within radius
        query = f"""
        [out:json][timeout:25];
        (
          way(around:{radius_m},{lat},{lng})[building];
          relation(around:{radius_m},{lat},{lng})[building];
        );
        out body geom;
        """

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.overpass_url,
                data={"data": query},
                headers={"User-Agent": self.user_agent},
            )
            resp.raise_for_status()
            data = resp.json()

        buildings = []
        for el in data.get("elements", []):
            geom = el.get("geometry", [])
            if len(geom) < 3:
                continue

            polygon = [[g["lat"], g["lon"]] for g in geom]
            lats = [p[0] for p in polygon]
            lngs = [p[1] for p in polygon]

            buildings.append({
                "polygon_ll": polygon,
                "bbox": [min(lats), min(lngs), max(lats), max(lngs)],
                "area_m2": polygon_area_m2(polygon),
                "source": "osm",
                "osm_id": f"{el['type']}/{el['id']}",
                "tags": el.get("tags", {}),
            })

        return buildings

    @staticmethod
    def _distance_to_centroid(lat: float, lng: float, polygon_ll: list[list[float]]) -> float:
        """Squared distance in degrees from (lat, lng) to polygon centroid."""
        clat = sum(p[0] for p in polygon_ll) / len(polygon_ll)
        clng = sum(p[1] for p in polygon_ll) / len(polygon_ll)
        return (clat - lat) ** 2 + (clng - lng) ** 2

    @staticmethod
    def _make_bbox_footprint(lat: float, lng: float, radius_m: float) -> dict:
        """Fallback: return a square bbox centered at (lat, lng)."""
        lat_half, lng_half = meters_to_degrees(radius_m, lat)
        polygon = [
            [lat - lat_half, lng - lng_half],
            [lat - lat_half, lng + lng_half],
            [lat + lat_half, lng + lng_half],
            [lat + lat_half, lng - lng_half],
            [lat - lat_half, lng - lng_half],
        ]
        return {
            "polygon_ll": polygon,
            "bbox": [lat - lat_half, lng - lng_half, lat + lat_half, lng + lng_half],
            "area_m2": (2 * radius_m) ** 2,
            "source": "bbox",
            "osm_id": None,
            "tags": {},
        }
