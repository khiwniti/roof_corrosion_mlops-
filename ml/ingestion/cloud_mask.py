"""Cloud masking with OmniCloudMask primary + 2-of-3 consensus fallback.

Architecture Decision: ADR-004
- Primary: OmniCloudMask v1.7 (Wright 2025)
- Secondary: CloudS2Mask / SCL
- Tertiary: s2cloudless
- Consensus: 2-of-3 before discarding a tile (protects bright zinc roofs)

SCL class codes used for quick pre-filter:
    3=cloud_shadow, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger("ingestion.cloud_mask")


def _scl_mask(scl: np.ndarray) -> np.ndarray:
    """SCL quick mask: True = clear, False = cloud/shadow."""
    # SCL classes: 3=cloud_shadow, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus
    bad = np.isin(scl, [3, 8, 9, 10])
    return ~bad


def _s2cloudless_mask(prob: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """s2cloudless GBM mask: True = clear."""
    return prob < threshold


def omnicloudmask_mask(
    rgb: np.ndarray,
    dynamic_nir: np.ndarray | None = None,
    bands_axis: int = -1,
) -> np.ndarray:
    """Run OmniCloudMask v1.7 on an RGB (or RGB+NIR) image.

    Parameters
    ----------
    rgb : (H, W, 3) uint8 or float array (R=Red, G=Green, B=Blue)
    dynamic_nir : optional (H, W) NIR band for better shadow detection
    bands_axis : axis containing bands if not last

    Returns
    -------
    mask : (H, W) bool array — True = clear, False = cloud/shadow
    """
    try:
        import omnicloudmask as ocm
    except ImportError:
        logger.error("omnicloudmask not installed; install with `pip install omnicloudmask`")
        raise

    # OmniCloudMask expects HWC float32 normalized 0-1
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if bands_axis != -1:
        rgb = np.moveaxis(rgb, bands_axis, -1)

    model = ocm.load_model()
    pred = ocm.predict(model, rgb, dynamic_nir=dynamic_nir)
    # OmniCloudMask returns classes:
    #   0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow
    # We want True = clear
    return pred == 0


def consensus_mask(
    scl: np.ndarray | None = None,
    s2cloudless_prob: np.ndarray | None = None,
    omnicloud_pred: np.ndarray | None = None,
    s2cloudless_threshold: float = 0.6,
) -> np.ndarray:
    """2-of-3 consensus cloud mask.

    Protects bright zinc roofs by not trusting any single mask alone.
    A pixel is marked CLEAR only if at least 2 of the 3 available masks say clear.

    Parameters
    ----------
    scl : SCL class array or None
    s2cloudless_prob : s2cloudless probability 0-1 or None
    omnicloud_pred : OmniCloudMask class array or None (0=clear)
    s2cloudless_threshold : probability threshold for s2cloudless

    Returns
    -------
    mask : bool array True = clear
    """
    masks: list[np.ndarray] = []

    if scl is not None:
        masks.append(_scl_mask(scl))
    if s2cloudless_prob is not None:
        masks.append(_s2cloudless_mask(s2cloudless_prob, s2cloudless_threshold))
    if omnicloud_pred is not None:
        masks.append(omnicloud_pred == 0)

    if len(masks) == 0:
        raise ValueError("No cloud masks provided")

    if len(masks) == 1:
        logger.warning("Only 1 cloud mask available; no consensus protection")
        return masks[0]

    # Majority vote: at least ceil(n/2) must agree clear
    # For 2 masks: both must agree (2-of-2)
    # For 3 masks: 2-of-3
    votes = np.stack(masks, axis=0).astype(np.uint8)
    clear_count = votes.sum(axis=0)
    required = (len(masks) + 1) // 2  # e.g. 2 for 3 masks, 1 for 2 masks
    # Actually for 2-of-3 we want >=2; for 2-of-2 we want >=2
    # For robustness with 2 masks, require both (2-of-2)
    return clear_count >= max(2, required)


def apply_cloud_mask(
    da: xr.DataArray,
    mask: np.ndarray,
) -> xr.DataArray:
    """Apply a 2-D boolean mask to a DataArray, setting masked pixels to NaN.

    Assumes the spatial dims are the last two dims of the DataArray.
    """
    # Ensure mask shape matches y, x
    y_dim, x_dim = da.dims[-2], da.dims[-1]
    if mask.shape != (da.sizes[y_dim], da.sizes[x_dim]):
        raise ValueError(
            f"Mask shape {mask.shape} does not match spatial dims "
            f"({da.sizes[y_dim]}, {da.sizes[x_dim]})"
        )

    # Broadcast mask to full DataArray shape
    mask_da = xr.DataArray(
        mask,
        dims=[y_dim, x_dim],
        coords={y_dim: da.coords[y_dim], x_dim: da.coords[x_dim]},
    )
    return da.where(mask_da)


def mask_stack(
    da: xr.DataArray,
    scl_stack: xr.DataArray | None = None,
    use_omnicloud: bool = True,
    use_scl: bool = True,
    use_s2cloudless: bool = False,
) -> xr.DataArray:
    """Apply cloud masking to a time-stacked S2 DataArray.

    Parameters
    ----------
    da : DataArray with dims (time, band, y, x)
    scl_stack : optional DataArray with SCL per time step
    use_omnicloud : run OmniCloudMask on RGB composite per time step
    use_scl : use SCL layer if available
    use_s2cloudless : use s2cloudless if available (requires S2CloudlessProb band)

    Returns
    -------
    DataArray with clouded pixels set to NaN.
    """
    if "time" not in da.dims:
        raise ValueError("Expected DataArray with 'time' dimension")

    masked = []
    for t in da.time:
        slice_t = da.sel(time=t)
        masks = []

        # SCL
        if use_scl and scl_stack is not None:
            scl = scl_stack.sel(time=t).values.squeeze()
            masks.append(_scl_mask(scl))

        # OmniCloudMask
        if use_omnicloud:
            # Extract RGB from B04, B03, B02
            try:
                r = slice_t.sel(band="B04").values
                g = slice_t.sel(band="B03").values
                b = slice_t.sel(band="B02").values
                rgb = np.stack([r, g, b], axis=-1)
                # Normalize to 0-1 (S2 L2A is 0-10000 DN)
                rgb = np.clip(rgb / 10000.0, 0, 1).astype(np.float32)
                masks.append(omnicloudmask_mask(rgb))
            except KeyError:
                logger.warning("RGB bands missing for OmniCloudMask; skipping")

        # s2cloudless
        if use_s2cloudless:
            try:
                prob = slice_t.sel(band="S2CloudlessProb").values.squeeze()
                masks.append(_s2cloudless_mask(prob))
            except KeyError:
                logger.warning("S2CloudlessProb band missing; skipping")

        if len(masks) == 0:
            logger.warning("No masks available for time=%s; keeping all pixels", t.values)
            masked.append(slice_t)
            continue

        cmask = consensus_mask(
            scl=masks[0] if (use_scl and scl_stack is not None) else None,
            omnicloud_pred=masks[1] if use_omnicloud and len(masks) > 1 else None,
            s2cloudless_prob=masks[-1] if use_s2cloudless else None,
        )
        masked.append(apply_cloud_mask(slice_t, cmask))

    return xr.concat(masked, dim="time")
