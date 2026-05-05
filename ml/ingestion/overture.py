"""Overture Maps building footprints for polygon AOI.

Architecture Decision: ADR-007
- Source: Overture Maps release (GeoParquet on S3)
- Filter: buildings theme, intersects polygon
- Output: GeoDataFrame with GERS ID, geometry, confidence, class
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

logger = logging.getLogger("ingestion.overture")

# Overture Maps GeoParquet S3 paths (release 2024-09-11 onwards)
OVERTURE_BASE = "s3://overturemaps-us-west-2/release"
OVERTURE_RELEASE = "2024-09-11"  # update periodically
BUILDINGS_PATH = (
    f"{OVERTURE_BASE}/{OVERTURE_RELEASE}/theme=buildings/type=building/*.parquet"
)


def fetch_buildings(
    polygon_geojson: dict[str, Any],
    release: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch Overture building footprints intersecting a polygon.

    Parameters
    ----------
    polygon_geojson : GeoJSON Polygon dict
    release : Overture release date (defaults to module constant)

    Returns
    -------
    GeoDataFrame with columns: id, geometry, confidence, class, height, num_floors
    """
    base = OVERTURE_BASE
    rel = release or OVERTURE_RELEASE
    path = f"{base}/{rel}/theme=buildings/type=building"

    aoi = shape(polygon_geojson)
    bbox = aoi.bounds  # (minx, miny, maxx, maxy)

    # Read only row groups that intersect bbox via pyarrow filtering
    # This requires pyarrow dataset API
    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        filter_expr = (
            (pc.field("bbox", "xmin") <= bbox[2])
            & (pc.field("bbox", "xmax") >= bbox[0])
            & (pc.field("bbox", "ymin") <= bbox[3])
            & (pc.field("bbox", "ymax") >= bbox[1])
        )
        table = dataset.to_table(filter=filter_expr)
        gdf = gpd.GeoDataFrame.from_features(table.to_pandas())
    except ImportError:
        logger.warning("pyarrow.dataset not available; using full scan (slow)")
        gdf = gpd.read_parquet(f"{path}/*.parquet")

    # Clip to exact polygon
    gdf = gdf[gdf.intersects(aoi)]
    gdf = gdf.copy()
    gdf["geometry"] = gdf.intersection(aoi)

    # Clean up columns
    keep_cols = ["id", "geometry", "confidence", "class"]
    if "height" in gdf.columns:
        keep_cols.append("height")
    if "num_floors" in gdf.columns:
        keep_cols.append("num_floors")

    gdf = gdf[[c for c in keep_cols if c in gdf.columns]]

    logger.info("Fetched %s buildings for AOI", len(gdf))
    return gdf


def compute_roof_area(
    gdf: gpd.GeoDataFrame,
    story_height_m: float = 3.5,
) -> dict[str, Any]:
    """Estimate total roof area from building footprints.

    Parameters
    ----------
    gdf : GeoDataFrame with building footprints
    story_height_m : average height per floor for area estimation

    Returns
    -------
    dict with building_count, total_area_m2, avg_area_m2, estimated_height_stats
    """
    if gdf.empty:
        return {
            "building_count": 0,
            "total_area_m2": 0.0,
            "avg_area_m2": 0.0,
            "height_stats": {},
        }

    areas = gdf.geometry.area
    total = areas.sum()
    avg = areas.mean()

    height_stats = {}
    if "height" in gdf.columns:
        heights = gdf["height"].dropna()
        if len(heights) > 0:
            height_stats = {
                "mean_m": float(heights.mean()),
                "min_m": float(heights.min()),
                "max_m": float(heights.max()),
            }

    return {
        "building_count": int(len(gdf)),
        "total_area_m2": float(total),
        "avg_area_m2": float(avg),
        "height_stats": height_stats,
    }
