"""Tests for ml/inference/tier0.py"""

import numpy as np
import pytest

from inference.tier0 import predict, estimate_roof_area


def test_predict_shape():
    features = np.random.rand(128, 128, 10).astype(np.float32)
    feature_names = ["ndvi", "B11", "B12", "B08", "B04", "iron_oxide", "B03", "B02", "B05", "B06"]
    result = predict(features, feature_names)

    assert result["material_mask"].shape == (128, 128)
    assert result["material_mask"].dtype == np.uint8
    assert set(np.unique(result["material_mask"])) <= {0, 1, 2, 3, 4}
    assert "corrosion_prob" in result
    assert "severity" in result


def test_predict_area_sums():
    features = np.random.rand(100, 100, 10).astype(np.float32)
    feature_names = ["ndvi", "B11", "B12", "B08", "B04", "iron_oxide", "B03", "B02", "B05", "B06"]
    result = predict(features, feature_names)

    total_px = 100 * 100
    total_class_px = sum(result["class_areas"].values())
    assert total_class_px == total_px

    total_pct = sum(result["class_percentages"].values())
    assert pytest.approx(total_pct, 1e-1) == 100.0

    # Corrosion stub should be proportional to material breakdown
    assert 0 <= result["corrosion_prob"] <= 1
    assert result["severity"] in ["none", "light", "moderate", "severe"]


def test_estimate_roof_area_defaults():
    mask = np.zeros((10, 10), dtype=np.uint8)
    area = estimate_roof_area(mask, gsd_m=5.0)
    assert area["roof_area_m2"] == 100 * 25
    assert area["pixel_area_m2"] == 25.0


def test_estimate_roof_area_with_buildings():
    mask = np.zeros((10, 10), dtype=np.uint8)
    area = estimate_roof_area(mask, gsd_m=5.0, building_count=4)
    assert area["building_count"] == 4
    assert area["avg_area_per_building_m2"] == 100 * 25 / 4
