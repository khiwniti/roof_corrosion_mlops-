"""Tests for ml/eval/cml_pr_eval.py"""

import pytest

from eval.cml_pr_eval import run_quick_benchmark, generate_report, save_cml_outputs


def test_run_quick_benchmark():
    metrics = run_quick_benchmark("stub-uri", max_samples=10)
    assert "material_mIoU" in metrics
    assert "corrosion_mIoU" in metrics
    assert "severity_accuracy" in metrics
    assert metrics["samples_evaluated"] == 10


def test_generate_report():
    metrics = {
        "material_mIoU": 0.67,
        "corrosion_mIoU": 0.51,
        "severity_accuracy": 0.73,
        "mean_inference_ms": 245.0,
        "samples_evaluated": 50,
    }
    report = generate_report(metrics, "mlflow:/test/1")

    assert "Material mIoU" in report
    assert "0.6700" in report
    assert "GO" in report or "CONDITIONAL" in report
    assert "CML-report" in report


def test_generate_report_with_baseline():
    metrics = {"material_mIoU": 0.67, "corrosion_mIoU": 0.51, "severity_accuracy": 0.73}
    baseline = {"material_mIoU": 0.65, "corrosion_mIoU": 0.49, "severity_accuracy": 0.71}
    report = generate_report(metrics, "mlflow:/test/1", baseline)

    assert "+0.0200" in report  # material delta
    assert "🟢" in report


def test_save_cml_outputs(tmp_path):
    metrics = {"material_mIoU": 0.67}
    report = "# Test Report"
    outputs = save_cml_outputs(report, metrics, str(tmp_path))

    assert outputs["report"].exists()
    assert outputs["metrics"].exists()
    assert outputs["report"].read_text() == report
