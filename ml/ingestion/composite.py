"""BAP (Best-Available-Pixel) compositing with fallback chain.

Architecture Decision: ADR-005
- Primary: 30-day BAP composite via openEO on CDSE
- Fallback: 60-d BAP → 90-d median → S1 SAR-aided gap fill → "deferred"

For Phase 1a we implement:
1. Median composite (fast, no openEO dependency)
2. BAP placeholder (openEO integration for production)
3. Coverage scoring to drive fallback chain
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger("ingestion.composite")


def _coverage_score(da: xr.DataArray) -> float:
    """Fraction of non-NaN pixels in a DataArray."""
    total = da.size
    valid = np.isfinite(da).sum().item()
    return valid / total if total > 0 else 0.0


def median_composite(
    da: xr.DataArray,
    min_coverage: float = 0.5,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Simple median composite across time dimension.

    Parameters
    ----------
    da : DataArray with dims (time, band, y, x) or (time, y, x)
    min_coverage : minimum valid pixel fraction to accept composite.

    Returns
    -------
    composite : median DataArray (band, y, x) or (y, x)
    meta : dict with coverage score and method
    """
    if "time" not in da.dims:
        raise ValueError("Expected DataArray with 'time' dimension")

    median = da.median(dim="time", skipna=True)
    coverage = _coverage_score(median)
    meta = {
        "method": "median",
        "coverage": coverage,
        "time_steps": int(da.sizes["time"]),
        "valid": coverage >= min_coverage,
    }
    logger.info("Median composite: coverage=%.2f, steps=%s", coverage, da.sizes["time"])
    return median, meta


def bap_composite(
    da: xr.DataArray,
    cloud_distance_weight: float = 0.3,
    coverage_weight: float = 0.5,
    date_proximity_weight: float = 0.2,
    reference_date: str | None = None,
    min_coverage: float = 0.5,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Best-Available-Pixel composite.

    Scores each time step per pixel:
        score = w1*dist_to_cloud + w2*coverage + w3*date_proximity

    For Phase 1a this is a simplified implementation using cloud-mask
    availability as proxy for distance-to-cloud. Production will use
    openEO BAP recipe on CDSE.

    Parameters
    ----------
    da : DataArray (time, band, y, x)
    cloud_distance_weight : weight for cloud distance (0-1)
    coverage_weight : weight for per-pixel coverage
    date_proximity_weight : weight for recency
    reference_date : target date (ISO string). Defaults to latest time.
    min_coverage : minimum valid pixel fraction.

    Returns
    -------
    composite, meta
    """
    if "time" not in da.dims:
        raise ValueError("Expected DataArray with 'time' dimension")

    n_time = da.sizes["time"]
    if n_time == 0:
        raise ValueError("No time steps available")

    # Compute per-pixel, per-time validity (clear = not NaN)
    valid = np.isfinite(da)
    if "band" in da.dims:
        # A pixel is valid if ALL bands are valid at that time
        valid = valid.all(dim="band")

    # Coverage score per time step: fraction of clear pixels
    coverage_per_t = valid.mean(dim=[d for d in valid.dims if d != "time"])

    # Date proximity: closer to reference_date = higher score
    if reference_date:
        ref = np.datetime64(reference_date)
    else:
        ref = da.time.max().values

    # Days from reference
    days = np.abs((da.time.values - ref).astype("timedelta64[D]").astype(float))
    max_days = days.max() if days.max() > 0 else 1.0
    date_score = 1.0 - (days / max_days)

    # Simple cloud distance proxy: valid neighbor ratio
    # For each time step, count valid pixels / total
    cloud_dist = coverage_per_t.values  # reuse coverage as proxy

    # Normalize weights
    total_w = cloud_distance_weight + coverage_weight + date_proximity_weight
    w1 = cloud_distance_weight / total_w
    w2 = coverage_weight / total_w
    w3 = date_proximity_weight / total_w

    # Per-time-step score
    scores = w1 * cloud_dist + w2 * cloud_dist + w3 * date_score
    best_t_idx = int(np.argmax(scores))

    # Build composite: for each pixel, pick the time with highest score
    # that has valid data
    composite = da.isel(time=best_t_idx).copy()

    # For pixels that are NaN in best time, fall back to median
    median = da.median(dim="time", skipna=True)
    composite = xr.where(np.isfinite(composite), composite, median)

    coverage = _coverage_score(composite)
    meta = {
        "method": "bap_simplified",
        "coverage": coverage,
        "time_steps": n_time,
        "best_time_index": best_t_idx,
        "best_time": str(da.time.values[best_t_idx]),
        "valid": coverage >= min_coverage,
        "weights": {"cloud_dist": w1, "coverage": w2, "date": w3},
    }
    logger.info(
        "BAP composite: coverage=%.2f, best_t=%s",
        coverage,
        meta["best_time"],
    )
    return composite, meta


def composite_with_fallback(
    da: xr.DataArray,
    lookback_days: int = 30,
    min_coverage: float = 0.5,
    reference_date: str | None = None,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Run composite with fallback chain: 30d BAP → 60d median → 90d median → deferred.

    Parameters
    ----------
    da : DataArray (time, band, y, x)
    lookback_days : initial lookback window
    min_coverage : minimum valid pixel fraction to accept
    reference_date : target date

    Returns
    -------
    composite, meta
    """
    # Try BAP first
    composite, meta = bap_composite(da, reference_date=reference_date, min_coverage=min_coverage)

    if meta["valid"]:
        meta["fallback_level"] = 0
        return composite, meta

    logger.warning("30d BAP coverage %.2f < %.2f; falling back to 60d median", meta["coverage"], min_coverage)

    # Fallback 1: 60d median (if more data available)
    # Note: if da only has 30d of data, this won't help; production would re-search
    composite, meta = median_composite(da, min_coverage=min_coverage)
    if meta["valid"]:
        meta["fallback_level"] = 1
        meta["method"] = "median_60d"
        return composite, meta

    logger.warning("60d median coverage %.2f < %.2f; falling back to 90d median", meta["coverage"], min_coverage)

    # Fallback 2: 90d median (same data, but flagged as deferred precision)
    composite, meta = median_composite(da, min_coverage=0.3)  # lower threshold
    meta["fallback_level"] = 2
    meta["method"] = "median_90d_deferred"
    meta["deferred_precision"] = True
    meta["valid"] = meta["coverage"] >= 0.3

    if not meta["valid"]:
        logger.error("All compositing failed. Coverage %.2f insufficient.", meta["coverage"])
        meta["status"] = "deferred"

    return composite, meta
