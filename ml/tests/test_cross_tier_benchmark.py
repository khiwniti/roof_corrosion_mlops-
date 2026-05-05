"""Tests for ml/eval/cross_tier_benchmark.py"""

import pytest

from eval.cross_tier_benchmark import run_tier0, run_tier1, run_tier3, compute_delta, generate_report


def test_run_tier0():
    result = run_tier0({})
    assert result["tier"] == 0
    assert "material_mIoU" in result
    assert "corrosion_prob" in result
    assert result["gsd_m"] == 10.0


def test_run_tier1():
    result = run_tier1({})
    assert result["tier"] == 1
    assert result["gsd_m"] == 0.5
    assert "confidence" in result


def test_run_tier3():
    result = run_tier3({})
    assert result["tier"] == 3
    assert result["gsd_m"] == 0.05
    assert "note" in result


def test_compute_delta():
    baseline = {"material_mIoU": 0.5, "corrosion_mIoU": 0.3}
    target = {"material_mIoU": 0.7, "corrosion_mIoU": 0.4}
    deltas = compute_delta(baseline, target)

    assert deltas["material_mIoU"]["absolute"] == 0.2
    assert deltas["corrosion_mIoU"]["absolute"] == 0.1


def test_generate_report():
    t0 = run_tier0({})
    t1 = run_tier1({})
    t3 = run_tier3({})
    report = generate_report({}, t0, t1, t3)

    assert "Cross-Tier Evaluation Benchmark" in report
    assert "Tier-0 (S2)" in report
    assert "Tier-1 (VHR)" in report
    assert "Tier-3 (Drone)" in report
