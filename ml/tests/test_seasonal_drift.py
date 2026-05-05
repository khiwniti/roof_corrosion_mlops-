"""Tests for ml/eval/seasonal_drift.py"""

import numpy as np
import pytest

from eval.seasonal_drift import (
    compute_context_features,
    compute_embeddings,
    run_seasonal_drift_check,
)


def test_compute_context_features():
    dates = ["2024-06-15T00:00:00Z", "2024-12-25T00:00:00Z"]
    provinces = ["Bangkok", "Chiang Mai"]
    ctx = compute_context_features(dates, provinces)

    assert ctx.shape == (2, 3)
    assert ctx[0, 0] == 6  # June
    assert ctx[1, 0] == 12  # December
    assert ctx[0, 1] == 167  # Day of year for June 15
    assert ctx[0, 2] == 0  # Bangkok (alphabetically first)
    assert ctx[1, 2] == 1  # Chiang Mai


def test_compute_embeddings_shape():
    images = [np.zeros((64, 64, 3)) for _ in range(5)]
    emb = compute_embeddings(images, encoder=None)

    assert emb.shape == (5, 512)
    # Check L2 normalization
    norms = np.linalg.norm(emb, axis=1)
    assert pytest.approx(norms.mean(), 1e-6) == 1.0


def test_run_seasonal_drift_check_stub():
    ref = {
        "images": [np.zeros((64, 64, 3)) for _ in range(10)],
        "dates": ["2024-01-01T00:00:00Z"] * 10,
        "provinces": ["Bangkok"] * 10,
    }
    prod = {
        "images": [np.zeros((64, 64, 3)) for _ in range(10)],
        "dates": ["2024-06-01T00:00:00Z"] * 10,
        "provinces": ["Bangkok"] * 10,
    }
    result = run_seasonal_drift_check(ref, prod, threshold=0.05)

    assert "drift_detected" in result
    assert "p_value" in result
    assert "method" in result
    assert result["status"] in ["ok", "alibi_detect_not_installed"]
