"""Tests for ml/ingestion/features.py"""

import numpy as np
import pytest

xarray = pytest.importorskip("xarray")

from ingestion.features import compute_indices, compute_glcm_textures, normalize_band


def test_normalize_band_basic():
    arr = np.array([0, 128, 255], dtype=np.float32)
    normalized = normalize_band(arr)
    assert pytest.approx(normalized.min(), 1e-6) == 0.0
    assert pytest.approx(normalized.max(), 1e-6) == 1.0


def test_compute_indices_shapes():
    bands = {
        "B04": np.random.rand(64, 64).astype(np.float32),
        "B08": np.random.rand(64, 64).astype(np.float32),
        "B03": np.random.rand(64, 64).astype(np.float32),
        "B02": np.random.rand(64, 64).astype(np.float32),
        "B11": np.random.rand(64, 64).astype(np.float32),
        "B12": np.random.rand(64, 64).astype(np.float32),
        "B05": np.random.rand(64, 64).astype(np.float32),
        "B06": np.random.rand(64, 64).astype(np.float32),
        "B07": np.random.rand(64, 64).astype(np.float32),
        "B8A": np.random.rand(64, 64).astype(np.float32),
    }
    indices = compute_indices(bands)

    assert "NDVI" in indices
    assert "NDWI" in indices
    assert "NDBI" in indices
    assert "NDMI" in indices
    assert "SAVI" in indices
    assert "NDRE" in indices
    assert "NDVI" in indices

    for key, arr in indices.items():
        assert arr.shape == (64, 64)
        assert np.isfinite(arr).all() or np.isnan(arr).sum() > 0


def test_compute_glcm_textures_shape():
    band = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
    textures = compute_glcm_textures(band)

    assert "contrast" in textures
    assert "dissimilarity" in textures
    assert "homogeneity" in textures
    assert "energy" in textures
    assert "correlation" in textures
    assert "ASM" in textures

    for key, arr in textures.items():
        assert arr.shape == (32, 32)
        assert np.isfinite(arr).all()
