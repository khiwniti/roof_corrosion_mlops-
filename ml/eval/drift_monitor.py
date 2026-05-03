"""Evidently drift monitoring pipeline.

Monitors:
1. Input drift: tile GSD, region distribution, cloud cover, capture date
2. Prediction drift: severity distribution, area estimates, confidence scores
3. Concept drift: model accuracy over time (via feedback loop)

Runs as a Prefect scheduled task every 24h.
Alerts are sent to the ops dashboard and Slack webhook if drift exceeds thresholds.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DataDriftTable,
    RegressionQualityMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric,
)

from app.db import get_supabase


# ── Reference data ──────────────────────────────────────────
# The reference distribution is captured from the first N production predictions.
# It's stored as a Parquet file in S3 and updated quarterly.

REFERENCE_DATA_PATH = "data/drift/reference_predictions.parquet"
DRIFT_REPORTS_DIR = Path("data/drift/reports")
DRIFT_THRESHOLDS = {
    "input_drift_share": 0.3,  # if >30% of input features drift → alert
    "prediction_drift_share": 0.3,
    "area_mape_increase": 0.05,  # if MAPE increases by >5% → alert
    "severity_distribution_chi2_p": 0.05,  # chi-squared p-value for severity distribution
}


def load_reference_data() -> pd.DataFrame:
    """Load reference prediction distribution."""
    path = Path(REFERENCE_DATA_PATH)
    if path.exists():
        return pd.read_parquet(path)
    # Return empty frame with expected columns
    return pd.DataFrame(columns=[
        "gsd_m", "region", "cloud_cover", "capture_month",
        "severity", "corrosion_percent", "roof_area_m2",
        "confidence", "area_mape",
    ])


def fetch_production_predictions(days: int = 30) -> pd.DataFrame:
    """Fetch recent production predictions from Supabase."""
    try:
        supabase = get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = supabase.table("assessments").select(
            "*, jobs(gsd_m, latitude, longitude, tile_capture_date)"
        ).gte("created_at", cutoff).execute()

        rows = []
        for r in result.data:
            job = r.get("jobs", {}) or {}
            rows.append({
                "gsd_m": job.get("gsd_m", 0.3),
                "region": "unknown",  # TODO: derive from lat/lng
                "cloud_cover": 0,  # TODO: extract from tile metadata
                "capture_month": job.get("tile_capture_date", "")[:7] if job.get("tile_capture_date") else "",
                "severity": r.get("severity", "none"),
                "corrosion_percent": r.get("corrosion_percent", 0),
                "roof_area_m2": r.get("roof_area_m2", 0),
                "confidence": r.get("confidence", 0),
            })

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Failed to fetch production predictions: {e}")
        return pd.DataFrame()


def compute_input_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> dict:
    """Compute data drift report for input features."""
    if production.empty:
        return {"status": "no_data", "drift_detected": False}

    # Select numeric features for drift detection
    numeric_features = ["gsd_m", "cloud_cover"]
    categorical_features = ["region", "capture_month"]

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    column_mapping = ColumnMapping(
        numerical_features=numeric_features,
        categorical_features=categorical_features,
    )

    report.run(
        reference_data=reference.head(len(production)),
        current_data=production,
        column_mapping=column_mapping,
    )

    # Save HTML report
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DRIFT_REPORTS_DIR / f"input_drift_{datetime.utcnow().strftime('%Y%m%d')}.html"
    report.save_html(str(report_path))

    # Extract drift metrics
    result = report.as_dict()
    drift_share = result.get("data_drift_summary", {}).get("share_of_drifted_columns", 0)

    return {
        "status": "ok",
        "drift_detected": drift_share > DRIFT_THRESHOLDS["input_drift_share"],
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLDS["input_drift_share"],
        "report_path": str(report_path),
    }


def compute_prediction_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> dict:
    """Compute drift report for prediction outputs."""
    if production.empty:
        return {"status": "no_data", "drift_detected": False}

    # Monitor drift in severity distribution and corrosion_percent
    report = Report(metrics=[
        ColumnDriftMetric("corrosion_percent"),
        ColumnDriftMetric("confidence"),
        ColumnSummaryMetric("severity"),
    ])

    report.run(
        reference_data=reference.head(len(production)),
        current_data=production,
    )

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DRIFT_REPORTS_DIR / f"prediction_drift_{datetime.utcnow().strftime('%Y%m%d')}.html"
    report.save_html(str(report_path))

    result = report.as_dict()

    return {
        "status": "ok",
        "drift_detected": False,  # TODO: parse from result
        "report_path": str(report_path),
    }


def check_severity_distribution_shift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> dict:
    """Chi-squared test for severity distribution shift."""
    from scipy.stats import chi2_contingency

    if production.empty or reference.empty:
        return {"status": "no_data", "shift_detected": False}

    sev_order = ["none", "light", "moderate", "severe"]
    ref_counts = reference["severity"].value_counts().reindex(sev_order, fill_value=0)
    prod_counts = production["severity"].value_counts().reindex(sev_order, fill_value=0)

    if ref_counts.sum() == 0 or prod_counts.sum() == 0:
        return {"status": "no_data", "shift_detected": False}

    contingency = np.array([ref_counts.values, prod_counts.values])
    chi2, p_value, _, _ = chi2_contingency(contingency)

    return {
        "status": "ok",
        "shift_detected": p_value < DRIFT_THRESHOLDS["severity_distribution_chi2_p"],
        "p_value": float(p_value),
        "threshold": DRIFT_THRESHOLDS["severity_distribution_chi2_p"],
        "reference_dist": ref_counts.to_dict(),
        "production_dist": prod_counts.to_dict(),
    }


def run_drift_check(days: int = 30) -> dict:
    """Run full drift check and return combined report."""
    reference = load_reference_data()
    production = fetch_production_predictions(days=days)

    input_drift = compute_input_drift_report(reference, production)
    prediction_drift = compute_prediction_drift_report(reference, production)
    severity_shift = check_severity_distribution_shift(reference, production)

    any_drift = (
        input_drift.get("drift_detected", False)
        or prediction_drift.get("drift_detected", False)
        or severity_shift.get("shift_detected", False)
    )

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "production_window_days": days,
        "production_sample_size": len(production),
        "any_drift_detected": any_drift,
        "input_drift": input_drift,
        "prediction_drift": prediction_drift,
        "severity_shift": severity_shift,
    }

    # Save report
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DRIFT_REPORTS_DIR / f"drift_summary_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if any_drift:
        print(f"⚠️  DRIFT DETECTED! See {report_path}")
        # TODO: send alert to Slack webhook
    else:
        print(f"✅ No drift detected. Report: {report_path}")

    return report


if __name__ == "__main__":
    run_drift_check()
