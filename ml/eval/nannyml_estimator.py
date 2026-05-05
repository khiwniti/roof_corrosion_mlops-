"""Label-free performance estimation using NannyML.

Architecture Decision: ADR-011
- Uses Confidence-Based Performance Estimation (CBPE) for regression targets
  (roof area m², corrosion percent)
- Uses Direct Loss Estimation (DLE) for classification targets
  (material class, severity)
- Runs daily on production predictions; alerts when estimated mIoU drops
  below threshold (e.g., 0.55 for Tier-0, 0.65 for Tier-1)

Install: pip install nannyml>=0.12.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("eval.nannyml_estimator")


def _make_synthetic_reference(
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic reference data for CBPE/DLE testing.

    In production, this would be loaded from labeled validation set.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "confidence": rng.uniform(0.3, 1.0, size=n),
        "predicted_severity": rng.choice(["none", "light", "moderate", "severe"], size=n),
        "predicted_material": rng.choice(["metal", "tile", "concrete"], size=n),
        "predicted_area_m2": rng.uniform(50, 500, size=n),
        "actual_severity": rng.choice(["none", "light", "moderate", "severe"], size=n),
        "actual_material": rng.choice(["metal", "tile", "concrete"], size=n),
        "actual_area_m2": rng.uniform(50, 500, size=n),
    })
    return df


def estimate_regression_performance(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    target_column: str = "predicted_area_m2",
    actual_column: str = "actual_area_m2",
    confidence_column: str = "confidence",
) -> dict[str, Any]:
    """Estimate regression performance (area MAE) using CBPE.

    NannyML CBPE requires calibrated confidence scores to estimate
    expected error. This stub computes a simple heuristic.

    Returns estimated MAE and confidence-weighted sample size.
    """
    try:
        import nannyml as nml
    except ImportError:
        logger.warning("nannyml not installed; returning heuristic estimate")
        # Heuristic: higher confidence → lower expected error
        expected_mae = production[confidence_column].apply(
            lambda c: 100 * (1 - c) if c < 0.7 else 30 * (1 - c)
        ).mean()
        return {
            "estimated_mae": round(float(expected_mae), 2),
            "method": "heuristic_confidence",
            "status": "nannyml_not_installed",
            "sample_size": len(production),
        }

    # Production CBPE setup
    cbpe = nml.CBPE(
        y_pred=target_column,
        y_pred_proba=confidence_column,
        y_true=actual_column,
        timestamp_column_name="timestamp",
        metrics=["mae"],
        chunk_period="d",
    )
    cbpe.fit(reference)
    results = cbpe.estimate(production)

    mae_values = results.filter(period="analysis").to_df()
    latest_mae = mae_values.iloc[-1]["mae"]
    alert = latest_mae > 50  # threshold: 50 m² area error

    return {
        "estimated_mae": round(float(latest_mae), 2),
        "method": "CBPE",
        "status": "ok",
        "alert": bool(alert),
        "sample_size": len(production),
        "reference_size": len(reference),
    }


def estimate_classification_performance(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    pred_column: str = "predicted_severity",
    actual_column: str = "actual_severity",
    confidence_column: str = "confidence",
) -> dict[str, Any]:
    """Estimate classification performance (severity accuracy) using DLE.

    Stub: uses confidence-weighted accuracy heuristic when NannyML
    is not available.
    """
    try:
        import nannyml as nml
    except ImportError:
        logger.warning("nannyml not installed; returning heuristic estimate")
        # Heuristic: weighted accuracy based on confidence
        weighted_acc = production[confidence_column].mean()
        return {
            "estimated_accuracy": round(float(weighted_acc), 2),
            "method": "heuristic_confidence",
            "status": "nannyml_not_installed",
            "sample_size": len(production),
        }

    # DLE for multiclass
    dle = nml.DLE(
        y_pred=pred_column,
        y_true=actual_column,
        timestamp_column_name="timestamp",
        metrics=["accuracy"],
        chunk_period="d",
    )
    dle.fit(reference)
    results = dle.estimate(production)

    acc_values = results.filter(period="analysis").to_df()
    latest_acc = acc_values.iloc[-1]["accuracy"]
    alert = latest_acc < 0.6  # threshold: 60% accuracy

    return {
        "estimated_accuracy": round(float(latest_acc), 2),
        "method": "DLE",
        "status": "ok",
        "alert": bool(alert),
        "sample_size": len(production),
        "reference_size": len(reference),
    }


def run_daily_performance_check(
    production_df: pd.DataFrame | None = None,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run full daily label-free performance estimation.

    Returns combined report with regression and classification estimates.
    """
    reference = reference_df if reference_df is not None else _make_synthetic_reference(n=1000)
    production = production_df if production_df is not None else _make_synthetic_reference(n=200, seed=43)

    regression = estimate_regression_performance(reference, production)
    classification = estimate_classification_performance(reference, production)

    any_alert = regression.get("alert", False) or classification.get("alert", False)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "any_alert": any_alert,
        "regression": regression,
        "classification": classification,
    }

    if any_alert:
        logger.warning("Performance alert triggered: %s", report)
    else:
        logger.info("Performance within expected bounds")

    return report
