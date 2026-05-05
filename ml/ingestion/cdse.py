"""CDSE Sentinel-2 ingestion via pystac-client + odc-stac.

Architecture Decision: ADR-003
- Catalog: pystac-client 0.9 → CDSE STAC API
- Loader: odc-stac 1.x (lazy load + clip to AOI)
- Always query L2A directly (CDSE generates on-the-fly from L1C if needed)
- Auth: OAuth2 client credentials (CDSE Keycloak)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pystac_client
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from odc.stac import load as odc_load
from shapely.geometry import Polygon, shape

logger = logging.getLogger("ingestion.cdse")

CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
THAILAND_MGRS_TILES = ["47P", "47Q", "47N", "48P", "48Q", "48N"]


def _get_auth_headers() -> dict[str, str] | None:
    """Return Authorization header with OAuth2 client-credentials token.

    Expects env vars CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.
    Returns None if credentials are missing (anonymous / public data only).
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.warning("CDSE_CLIENT_ID/SECRET not set; using anonymous access")
        return None

    import urllib.request
    import urllib.parse

    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        CDSE_AUTH_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json

            token = json.loads(resp.read().decode("utf-8"))
            access_token = token.get("access_token")
            if access_token:
                return {"Authorization": f"Bearer {access_token}"}
    except Exception as e:
        logger.warning("CDSE auth failed: %s", e)
    return None


def open_catalog() -> pystac_client.Client:
    """Open CDSE STAC catalog with mandatory conformance hint."""
    headers = _get_auth_headers()
    cat = pystac_client.Client.open(CDSE_STAC_URL, headers=headers)
    cat.add_conforms_to("ITEM_SEARCH")
    return cat


def search_s2_l2a(
    polygon_geojson: dict[str, Any],
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    max_cloud: int = 50,
    limit: int = 50,
) -> pystac_client.ItemSearch:
    """Search CDSE for Sentinel-2 L2A scenes overlapping a polygon.

    Parameters
    ----------
    polygon_geojson : GeoJSON Polygon dict
    start_date, end_date : ISO date strings or datetime. Defaults to last 90 days.
    max_cloud : max S2Cloudless probability (0–100). Default 50.
    limit : max items returned.

    Returns
    -------
    pystac_client.ItemSearch
    """
    cat = open_catalog()
    aoi = shape(polygon_geojson)

    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    date_range = f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z"

    search = cat.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    logger.info(
        "S2 L2A search: %s items, %s, cloud<%s",
        search.matched(),
        date_range,
        max_cloud,
    )
    return search


def fetch_s2_stack(
    search: pystac_client.ItemSearch,
    polygon_geojson: dict[str, Any],
    bands: list[str] | None = None,
    resolution: int = 10,
    dtype: str = "uint16",
    progress: bool = False,
) -> xr.DataArray:
    """Lazy-load S2 stack via odc-stac, clipped to polygon bbox.

    Parameters
    ----------
    search : pystac_client.ItemSearch
    polygon_geojson : GeoJSON Polygon dict (used for bbox & final clip)
    bands : S2 band names, e.g. ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
    resolution : target GSD in metres.
    dtype : output dtype.
    progress : show tqdm bar.

    Returns
    -------
    xarray.DataArray with dims (time, band, y, x)
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    items = search.item_collection()
    if len(items) == 0:
        raise ValueError("No S2 L2A items found for the given criteria")

    aoi = shape(polygon_geojson)
    bbox = aoi.bounds  # (minx, miny, maxx, maxy)

    logger.info("Loading %s items, bands=%s, res=%sm", len(items), bands, resolution)

    ds = odc_load(
        items,
        bands=bands,
        resolution=resolution,
        crs="EPSG:4326",
        bbox=bbox,
        dtype=dtype,
        progress=progress,
        groupby="solar_day",
    )

    # Clip to exact polygon
    da = ds.to_array(dim="band") if hasattr(ds, "to_array") else ds
    da = da.rio.clip([polygon_geojson], crs="EPSG:4326")

    return da


def fetch_s2_single_composite(
    polygon_geojson: dict[str, Any],
    reference_date: datetime | None = None,
    lookback_days: int = 30,
    max_cloud: int = 50,
    resolution: int = 10,
) -> xr.DataArray:
    """High-level helper: search → load stack for a polygon.

    Returns an xarray DataArray with dims (time, band, y, x).
    Caller is responsible for compositing (see composite.py).
    """
    end = reference_date or datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    search = search_s2_l2a(
        polygon_geojson,
        start_date=start,
        end_date=end,
        max_cloud=max_cloud,
        limit=20,
    )
    return fetch_s2_stack(
        search,
        polygon_geojson,
        resolution=resolution,
        progress=True,
    )


def compute_spectral_indices(
    da: xr.DataArray,
    band_coords: list[str],
) -> xr.DataArray:
    """Compute spectral indices from an S2 DataArray.

    Parameters
    ----------
    da : DataArray with dims (time, band, y, x) or (band, y, x)
    band_coords : list of band names matching da.coords["band"]

    Returns
    -------
    DataArray with new bands appended for indices.
    """
    def band(name: str) -> xr.DataArray:
        return da.sel(band=name)

    def _idx(name: str, expr: xr.DataArray) -> xr.DataArray:
        return expr.expand_dims(band=[name]).assign_coords(band=[name])

    idx_list = []

    # Iron Oxide (B4/B2)
    idx_list.append(_idx("iron_oxide", band("B04") / band("B02")))
    # Ferrous (B11/B8A)
    idx_list.append(_idx("ferrous", band("B11") / band("B8A")))
    # Iron Mixture ((B6+B7)/B8A)
    idx_list.append(_idx("iron_mixture", (band("B06") + band("B07")) / band("B8A")))
    # Clay minerals (B11/B12)
    idx_list.append(_idx("clay_minerals", band("B11") / band("B12")))
    # NDMCI port (B8A-B5)/(B8A+B5)
    b8a = band("B8A").astype("float32")
    b5 = band("B05").astype("float32")
    idx_list.append(_idx("ndmci", (b8a - b5) / (b8a + b5 + 1e-6)))
    # BCCSR (B4-B2)/(B4+B2)
    b4 = band("B04").astype("float32")
    b2 = band("B02").astype("float32")
    idx_list.append(_idx("bccsr", (b4 - b2) / (b4 + b2 + 1e-6)))
    # NDVI
    b8 = band("B08").astype("float32")
    idx_list.append(_idx("ndvi", (b8 - b4) / (b8 + b4 + 1e-6)))

    idx_da = xr.concat(idx_list, dim="band")
    return xr.concat([da, idx_da], dim="band")
