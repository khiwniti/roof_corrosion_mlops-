"""Tests for ml/inference/tier1.py"""

import numpy as np
import pytest
import torch

from inference.tier1 import (
    detect_buildings,
    segment_roof_sam2,
    temperature_scaling,
    predict,
)


def test_detect_buildings_stub():
    img = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
    buildings = detect_buildings(img, confidence=0.5)

    assert isinstance(buildings, list)
    assert len(buildings) > 0
    assert "bbox" in buildings[0]
    assert "score" in buildings[0]


def test_segment_roof_sam2_stub():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    bbox = [100, 100, 400, 400]
    mask = segment_roof_sam2(img, bbox)

    assert mask.shape == (512, 512)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_temperature_scaling():
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    probs = temperature_scaling(logits, temperature=2.0)

    assert probs.shape == (1, 3)
    assert pytest.approx(probs.sum().item(), 1e-6) == 1.0

    # Higher temperature = more uniform
    probs_high_temp = temperature_scaling(logits, temperature=10.0)
    probs_low_temp = temperature_scaling(logits, temperature=0.5)
    assert probs_high_temp.max() < probs_low_temp.max()


def test_predict_stub():
    vhr_img = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
    result = predict(vhr_img, model=None, temperature=1.0)

    assert "buildings" in result
    assert "building_count" in result
    assert "total_roof_area_px" in result
    assert "material_breakdown" in result
    assert "corrosion_detected" in result
    assert "temperature" in result
    assert "confidence" in result

    # Check material breakdown sums to ~100
    breakdown = result["material_breakdown"]
    total_pct = sum([
        breakdown["metal_percent"],
        breakdown["tile_percent"],
        breakdown["concrete_percent"],
        breakdown["vegetation_percent"],
        breakdown["other_percent"],
    ])
    assert pytest.approx(total_pct, 1) == 100.0
