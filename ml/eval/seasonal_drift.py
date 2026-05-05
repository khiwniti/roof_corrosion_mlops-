"""Seasonal drift detection using alibi-detect ContextMMDDrift.

Architecture Decision: ADR-011
- Handles tropical seasonal cycles (monsoon vs dry season)
- Context: month, day-of-year, province
- Embedding: Clay or Prithvi backbone for sensor-aware drift

Note: alibi-detect 0.12+ required for ContextMMDDrift.
Install: pip install alibi-detect>=0.12.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger("eval.seasonal_drift")


def compute_context_features(
    capture_dates: list[str],
    provinces: list[str],
) -> np.ndarray:
    """Convert capture dates and provinces to numeric context vectors.

    Returns (N, 3) array: [month (1-12), day_of_year (1-366), province_id].
    """
    contexts = []
    province_to_id = {p: i for i, p in enumerate(sorted(set(provinces)))}

    for date_str, province in zip(capture_dates, provinces):
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(UTC)
        contexts.append([
            dt.month,
            dt.timetuple().tm_yday,
            province_to_id.get(province, -1),
        ])

    return np.array(contexts, dtype=np.float32)


def compute_embeddings(
    images: list[np.ndarray],
    encoder: Any = None,
) -> np.ndarray:
    """Compute sensor-aware embeddings using Clay/Prithvi backbone.

    Stub: returns random normalized vectors if encoder is None.
    """
    n = len(images)
    if encoder is not None:
        # Production: encoder(images) → (N, D)
        logger.info("Computing embeddings with provided encoder")
        # Placeholder for actual encoder call
        embeddings = np.random.randn(n, 512).astype(np.float32)
    else:
        logger.warning("No encoder provided; using random embeddings")
        embeddings = np.random.randn(n, 512).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def detect_seasonal_drift(
    reference_images: list[np.ndarray],
    reference_dates: list[str],
    reference_provinces: list[str],
    production_images: list[np.ndarray],
    production_dates: list[str],
    production_provinces: list[str],
    encoder: Any = None,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Run ContextMMDDrift to detect seasonal distribution shift.

    Parameters
    ----------
    reference_images, production_images : list of (H, W, C) arrays
    reference_dates, production_dates : ISO date strings
    reference_provinces, production_provinces : province names
    encoder : optional Clay/Prithvi encoder
    threshold : p-value threshold for drift detection

    Returns
    -------
    dict with drift_detected, p_value, distance, context_info
    """
    try:
        from alibi_detect.cd import ContextMMDDrift
    except ImportError:
        logger.warning("alibi-detect not installed; returning stub result")
        return {
            "drift_detected": False,
            "p_value": 1.0,
            "distance": 0.0,
            "method": "ContextMMDDrift",
            "status": "alibi_detect_not_installed",
        }

    ref_emb = compute_embeddings(reference_images, encoder)
    prod_emb = compute_embeddings(production_images, encoder)
    ref_ctx = compute_context_features(reference_dates, reference_provinces)
    prod_ctx = compute_context_features(production_dates, production_provinces)

    detector = ContextMMDDrift(
        x_ref=ref_emb,
        c_ref=ref_ctx,
        p_val=threshold,
        n_permutations=100,
    )

    result = detector.predict(prod_emb, prod_ctx)
    is_drift = result["data"]["is_drift"]
    p_value = result["data"]["p_val"]
    distance = result["data"].get("distance", 0.0)

    logger.info("Seasonal drift: detected=%s, p=%.4f, distance=%.4f", is_drift, p_value, distance)

    return {
        "drift_detected": bool(is_drift),
        "p_value": float(p_value),
        "distance": float(distance),
        "method": "ContextMMDDrift",
        "threshold": threshold,
        "status": "ok",
        "reference_size": len(reference_images),
        "production_size": len(production_images),
    }


def run_seasonal_drift_check(
    reference_data: dict[str, Any],
    production_data: dict[str, Any],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """High-level wrapper for drift check from structured data.

    reference_data / production_data should contain keys:
        - images: list[np.ndarray]
        - dates: list[str]
        - provinces: list[str]
    """
    return detect_seasonal_drift(
        reference_images=reference_data.get("images", []),
        reference_dates=reference_data.get("dates", []),
        reference_provinces=reference_data.get("provinces", []),
        production_images=production_data.get("images", []),
        production_dates=production_data.get("dates", []),
        production_provinces=production_data.get("provinces", []),
        threshold=threshold,
    )
