"""Tests for ml/eval/nannyml_estimator.py"""

import pandas as pd
import pytest

from eval.nannyml_estimator import (
    estimate_regression_performance,
    estimate_classification_performance,
    run_daily_performance_check,
)


def test_estimate_regression_performance_heuristic():
    ref = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "confidence": [0.8] * 100,
        "predicted_area_m2": [100.0] * 100,
        "actual_area_m2": [100.0] * 100,
    })
    prod = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=20, freq="h"),
        "confidence": [0.5] * 20,
        "predicted_area_m2": [100.0] * 20,
        "actual_area_m2": [100.0] * 20,
    })
    result = estimate_regression_performance(ref, prod)

    assert "estimated_mae" in result
    assert result["method"] in ["CBPE", "heuristic_confidence"]
    assert "sample_size" in result


def test_estimate_classification_performance_heuristic():
    ref = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "confidence": [0.8] * 100,
        "predicted_severity": ["light"] * 100,
        "actual_severity": ["light"] * 100,
    })
    prod = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=20, freq="h"),
        "confidence": [0.5] * 20,
        "predicted_severity": ["moderate"] * 20,
        "actual_severity": ["moderate"] * 20,
    })
    result = estimate_classification_performance(ref, prod)

    assert "estimated_accuracy" in result
    assert result["method"] in ["DLE", "heuristic_confidence"]
    assert "sample_size" in result


def test_run_daily_performance_check():
    report = run_daily_performance_check()

    assert "timestamp" in report
    assert "regression" in report
    assert "classification" in report
    assert "any_alert" in report
    assert isinstance(report["any_alert"], bool)
