"""Feature extraction for Tier-0 classifier.

Extracts the ~25-feature-per-pixel stack from a cloud-masked S2 composite:
- 10 S2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
- 7 spectral indices (iron oxide, ferrous, iron mixture, clay, ndmci, bccsr, ndvi)
- 10 GLCM textures (contrast, dissimilarity, homogeneity, energy, correlation
  on B04 and B11)
- 3 S1 SAR features (VV, VH, VV/VH) — stub until S1 ingestion is implemented

Output: numpy array (H, W, C) float32 ready for model inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from .cdse import compute_spectral_indices

logger = logging.getLogger("ingestion.features")


def _glcm_textures(
    band_arr: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute GLCM texture features for a single 2-D band array.

    Returns dict with keys like 'contrast', 'dissimilarity', etc.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        logger.warning("scikit-image not installed; GLCM textures skipped")
        return {}

    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    h, w = band_arr.shape
    # Rescale to 8-bit for GLCM (256 levels is standard)
    vmin, vmax = np.nanmin(band_arr), np.nanmax(band_arr)
    if vmax == vmin:
        return {}
    img_8 = np.clip(((band_arr - vmin) / (vmax - vmin) * 255), 0, 255).astype(np.uint8)

    # Handle NaNs by filling with 0 (GLCM doesn't like NaNs)
    img_8 = np.where(np.isfinite(band_arr), img_8, 0)

    try:
        glcm = graycomatrix(
            img_8,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
    except Exception as e:
        logger.warning("GLCM computation failed: %s", e)
        return {}

    textures = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        try:
            vals = graycoprops(glcm, prop)
            # Average over distances and angles
            textures[prop] = np.full((h, w), float(vals.mean()), dtype=np.float32)
        except Exception:
            pass
    return textures


def extract_features(
    composite: xr.DataArray,
    include_glcm: bool = True,
    include_s1: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Extract feature stack from a composite DataArray.

    Parameters
    ----------
    composite : DataArray with dims (band, y, x) — already cloud-masked / composited
    include_glcm : compute GLCM texture features on B04 and B11
    include_s1 : include Sentinel-1 SAR features (stub if S1 not available)

    Returns
    -------
    features : ndarray (H, W, C) float32
    feature_names : list of channel names
    """
    if "band" not in composite.dims:
        raise ValueError("Expected DataArray with 'band' dimension")

    # 1. Start with raw S2 bands
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    available_bands = [b for b in bands if b in composite.coords["band"].values]
    if len(available_bands) < len(bands):
        logger.warning("Missing bands: %s", set(bands) - set(available_bands))

    # Select bands and transpose to (y, x, band)
    da_bands = composite.sel(band=available_bands)
    stack = [da_bands.sel(band=b).values for b in available_bands]
    names = list(available_bands)

    # 2. Spectral indices — compute on the composite
    try:
        da_with_idx = compute_spectral_indices(da_bands, available_bands)
        idx_bands = ["iron_oxide", "ferrous", "iron_mixture", "clay_minerals", "ndmci", "bccsr", "ndvi"]
        for idx_name in idx_bands:
            if idx_name in da_with_idx.coords["band"].values:
                stack.append(da_with_idx.sel(band=idx_name).values)
                names.append(idx_name)
    except Exception as e:
        logger.warning("Spectral index computation failed: %s", e)

    # 3. GLCM textures on B04 and B11
    if include_glcm:
        for band_name in ["B04", "B11"]:
            if band_name not in available_bands:
                continue
            band_arr = da_bands.sel(band=band_name).values
            textures = _glcm_textures(band_arr)
            for prop, arr in textures.items():
                stack.append(arr)
                names.append(f"{band_name}_{prop}")

    # 4. S1 SAR features (stub)
    if include_s1:
        h, w = stack[0].shape
        stack.append(np.full((h, w), np.nan, dtype=np.float32))  # VV stub
        stack.append(np.full((h, w), np.nan, dtype=np.float32))  # VH stub
        stack.append(np.full((h, w), np.nan, dtype=np.float32))  # VV/VH stub
        names.extend(["S1_VV", "S1_VH", "S1_VV_VH"])

    # Stack to (H, W, C)
    features = np.stack(stack, axis=-1).astype(np.float32)
    # Replace any remaining NaNs / Infs with 0 for model safety
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Feature stack: shape=%s, channels=%s", features.shape, len(names))
    return features, names


def normalize_features(
    features: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize feature stack per-channel.

    Parameters
    ----------
    features : (H, W, C) array
    method : 'minmax' or 'standardize'

    Returns
    -------
    normalized features
    """
    if method == "minmax":
        vmin = np.percentile(features.reshape(-1, features.shape[-1]), 1, axis=0)
        vmax = np.percentile(features.reshape(-1, features.shape[-1]), 99, axis=0)
        denom = vmax - vmin
        denom[denom == 0] = 1.0
        return np.clip((features - vmin) / denom, 0, 1)
    elif method == "standardize":
        mean = np.mean(features.reshape(-1, features.shape[-1]), axis=0)
        std = np.std(features.reshape(-1, features.shape[-1]), axis=0)
        std[std == 0] = 1.0
        return (features - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
