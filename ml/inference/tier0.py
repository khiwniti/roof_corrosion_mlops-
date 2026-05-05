"""Tier-0 inference: material segmentation + area estimation.

Phase 1a stub: returns plausible material masks using rule-based heuristics
on the feature stack. Replace with smp U-Net + EfficientNet-B7 once trained.

Architecture Decision: ADR-002 (superseded two-stage with multi-task;
Tier-0 uses a simplified 3-class material head: metal, tile, other).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("inference.tier0")

# Tier-0 material class mapping
CLASS_NAMES = ["metal", "tile", "concrete", "vegetation", "other"]
MATERIAL_CLASSES = ["metal", "tile", "other"]  # coarse grouping for UI


def _rule_based_mask(features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Rule-based material classification from spectral features.

    Uses simple thresholds on NIR, SWIR, and spectral indices to guess
    material type. This is a stub — real model will be smp U-Net.

    Returns class indices (H, W) where:
        0 = metal, 1 = tile, 2 = concrete, 3 = vegetation, 4 = other
    """
    h, w, c = features.shape
    # Try to find relevant feature channels by name
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Default to random plausible values if features missing
    def _ch(name: str) -> np.ndarray:
        if name in name_to_idx:
            return features[..., name_to_idx[name]]
        return np.zeros((h, w), dtype=np.float32)

    # Heuristic thresholds (very coarse — stub only)
    ndvi = _ch("ndvi")
    b11 = _ch("B11")
    b12 = _ch("B12")
    b8 = _ch("B08")
    b4 = _ch("B04")
    iron = _ch("iron_oxide")

    # Vegetation = high NDVI
    veg = ndvi > 0.4

    # Metal roofs: high SWIR reflectance (B11, B12), low NDVI, high iron oxide ratio
    metal = (b11 > 0.25) & (b12 > 0.20) & (ndvi < 0.2) & (iron > 1.5) & ~veg

    # Tile roofs: moderate SWIR, moderate NDVI, reddish (high B4)
    tile = (b4 > 0.25) & (b11 > 0.15) & (ndvi < 0.35) & ~metal & ~veg

    # Concrete: moderate everything, lower SWIR than metal
    concrete = (b11 > 0.10) & (b11 < 0.25) & ~metal & ~tile & ~veg

    # Remaining = other
    mask = np.full((h, w), 4, dtype=np.uint8)  # other
    mask[veg] = 3
    mask[concrete] = 2
    mask[tile] = 1
    mask[metal] = 0

    # Add some spatial smoothing (median filter)
    try:
        from scipy.ndimage import median_filter
        mask = median_filter(mask, size=5)
    except Exception:
        pass

    return mask


def predict(
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Run Tier-0 material segmentation.

    Parameters
    ----------
    features : (H, W, C) float32 array from extract_features
    feature_names : list of channel names

    Returns
    -------
    dict with:
        - material_mask: (H, W) uint8 class indices
        - class_names: list of class names
        - class_areas: dict of pixel counts per class
        - class_percentages: dict of percentages
        - confidence: float (stub — always 0.5 for now)
    """
    mask = _rule_based_mask(features, feature_names)

    total = mask.size
    class_areas = {}
    class_percentages = {}
    for i, name in enumerate(CLASS_NAMES):
        count = int((mask == i).sum())
        class_areas[name] = count
        class_percentages[name] = round(count / total * 100, 1) if total > 0 else 0.0

    # Coarse material grouping for Tier-0 UI
    metal_pct = class_percentages.get("metal", 0.0)
    tile_pct = class_percentages.get("tile", 0.0)
    other_pct = 100.0 - metal_pct - tile_pct

    # Stub corrosion prediction (heuristic: metal roofs in tropical climates
    # have higher corrosion probability)
    corrosion_prob = 0.25 if metal_pct > tile_pct else 0.15
    severity = "none"
    if corrosion_prob > 0.3:
        severity = "moderate"
    elif corrosion_prob > 0.15:
        severity = "light"

    corroded_px = int(mask.size * corrosion_prob)

    return {
        "material_mask": mask,
        "class_names": CLASS_NAMES,
        "class_areas": class_areas,
        "class_percentages": class_percentages,
        "coarse_breakdown": {
            "metal_percent": metal_pct,
            "tile_percent": tile_pct,
            "other_percent": max(0.0, other_pct),
        },
        "corrosion_prob": round(corrosion_prob, 2),
        "severity": severity,
        "corroded_area_px": corroded_px,
        "confidence": 0.5,  # stub: Tier-0 is always ±30%
        "model_version": "tier0-stub-v0",
    }


def estimate_roof_area(
    material_mask: np.ndarray,
    gsd_m: float = 10.0,
    building_count: int | None = None,
) -> dict[str, Any]:
    """Estimate total roof area from material mask and GSD.

    Parameters
    ----------
    material_mask : (H, W) class indices — 3=vegetation excluded
    gsd_m : ground sample distance in metres per pixel
    building_count : optional building count from Overture

    Returns
    -------
    dict with area_m2, building_count, avg_area_per_building_m2
    """
    pixel_area = gsd_m * gsd_m
    # Exclude vegetation class (3) from roof area
    roof_pixels = int((material_mask != 3).sum())
    area_m2 = roof_pixels * pixel_area

    count = building_count or 0
    avg = area_m2 / count if count > 0 else None

    return {
        "roof_area_m2": round(area_m2, 1),
        "building_count": count,
        "avg_area_per_building_m2": round(avg, 1) if avg else None,
        "pixel_area_m2": pixel_area,
    }
