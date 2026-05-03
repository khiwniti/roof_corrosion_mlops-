"""Foundation-Model pipeline — zero-training roof corrosion analysis.

Combines three foundation models / free services:
1. OSM Building Footprints → roof polygon (no model, free)
2. SAM 2                   → roof mask fallback when OSM misaligned
3. NVIDIA NIM VLM          → corrosion severity, material, rationale

This pipeline is the default production path. It replaces the two-stage
SegFormer pipeline for the MVP since:
- Zero training required
- Zero labeled data required
- Ships in days, not months
- VLM already understands "corrosion", "rust", "metal degradation"
- Costs ~$0.01-0.05 per quote — negligible vs quote value

The trained SegFormer pipeline is still available as a fallback / for
shadow-deploy comparison once real training data becomes available.

Usage (async):
    pipeline = FoundationModelPipeline()
    result = await pipeline.analyze(
        tile_image=np.array(...),
        lat=40.7128,
        lng=-74.0060,
        gsd=0.3,
        address="123 Main St",
    )
    # → CorrosionResult
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.inference.footprint_client import BuildingFootprintClient, polygon_area_m2
from app.inference.nim_client import NIMError, NIMVisionClient
from app.inference.types import CorrosionResult, classify_severity
from app.inference.sam_client import SAMClient

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

@dataclass
class FMPipelineConfig:
    """Configuration for the foundation-model pipeline."""
    use_osm_footprint: bool = True
    use_sam_fallback: bool = True
    sam_always: bool = False           # run SAM even when OSM has a polygon
    min_confidence: float = 0.6        # below this → requires_human_review
    vlm_max_image_size: int = 1024
    footprint_radius_m: float = 50.0


class FoundationModelPipeline:
    """Zero-training roof corrosion pipeline using VLM + SAM + OSM."""

    def __init__(
        self,
        nim_client: Optional[NIMVisionClient] = None,
        footprint_client: Optional[BuildingFootprintClient] = None,
        sam_client: Optional[SAMClient] = None,
        config: Optional[FMPipelineConfig] = None,
    ):
        # Lazy-init: only create NIM client if we actually use it
        self._nim = nim_client
        self.footprint = footprint_client or BuildingFootprintClient()
        self._sam = sam_client
        self.config = config or FMPipelineConfig()

        self.roof_model_uri = "osm-overpass + sam2"
        self.corrosion_model_uri = "nvidia-nim/llama-3.2-90b-vision-instruct"

    @property
    def nim(self) -> NIMVisionClient:
        if self._nim is None:
            self._nim = NIMVisionClient()
        return self._nim

    @property
    def sam(self) -> SAMClient:
        if self._sam is None:
            self._sam = SAMClient()
        return self._sam

    async def analyze(
        self,
        tile_image: np.ndarray,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        gsd: float = 0.3,
        address: str = "",
        tile_bounds: Optional[tuple[float, float, float, float]] = None,
    ) -> CorrosionResult:
        """Run the foundation-model pipeline end-to-end.

        Args:
            tile_image: (H, W, 3) uint8 RGB overhead image
            lat, lng: optional — WGS84 coords of the roof (for OSM + prompt)
            gsd: ground sample distance in meters/pixel
            address: optional — address string for VLM regional context
            tile_bounds: optional — (min_lat, min_lng, max_lat, max_lng) bounds
                         of the tile, used to project lat/lng into pixel coords

        Returns:
            CorrosionResult with roof mask, corrosion metrics, and model versions
        """
        h, w = tile_image.shape[:2]
        t_start = time.time()

        # ── 1. Roof mask ─────────────────────────────────────────
        roof_mask, footprint_info = await self._get_roof_mask(
            tile_image, lat, lng, tile_bounds
        )
        t_roof = time.time() - t_start
        logger.info(f"Roof mask: source={footprint_info.get('source')}, "
                    f"coverage={roof_mask.mean():.2%}, time={t_roof:.2f}s")

        # ── 2. VLM corrosion assessment ──────────────────────────
        # Crop tile to roof bbox (+ small padding) before sending to VLM
        roof_tile, roof_bbox = self._crop_to_roof(tile_image, roof_mask)
        roof_material_hint = footprint_info.get("tags", {}).get("building:material", "")

        t_vlm_start = time.time()
        try:
            vlm_result = await self.nim.assess_corrosion(
                tile_image=roof_tile,
                gsd=gsd,
                address=address,
                roof_context=f"material hint: {roof_material_hint}" if roof_material_hint else "",
                max_image_size=self.config.vlm_max_image_size,
            )
        except NIMError as e:
            logger.error(f"VLM assessment failed: {e}")
            # Degrade gracefully — return uncertain result
            vlm_result = {
                "assessable": False,
                "corrosion_percent": 0.0,
                "severity": "none",
                "confidence": 0.0,
                "description": f"VLM error: {e}",
                "rationale": "",
                "roof_material": "unknown",
                "visible_issues": [],
            }
        t_vlm = time.time() - t_vlm_start
        logger.info(f"VLM assessment: severity={vlm_result['severity']}, "
                    f"corrosion={vlm_result['corrosion_percent']:.1f}%, "
                    f"confidence={vlm_result['confidence']:.2f}, time={t_vlm:.2f}s")

        # ── 3. Compute areas from roof mask + VLM percent ────────
        roof_pixels = int(roof_mask.sum())
        roof_area_m2 = roof_pixels * gsd * gsd
        # If OSM gave us a real polygon area, prefer that (more accurate than pixel count)
        if footprint_info.get("source") == "osm" and footprint_info.get("area_m2", 0) > 0:
            roof_area_m2 = footprint_info["area_m2"]

        corrosion_percent = vlm_result["corrosion_percent"]
        corroded_area_m2 = roof_area_m2 * (corrosion_percent / 100.0)

        # Derive corrosion mask as a weighted roof mask (we don't have pixel-level
        # corrosion from VLM — use confidence-weighted roof mask as a proxy).
        # Future: use Grounding-DINO + SAM for actual corrosion pixels.
        corrosion_mask = np.zeros_like(roof_mask, dtype=bool)
        if corrosion_percent > 0 and roof_pixels > 0:
            # Randomly sample corrosion_percent of roof pixels (deterministic seed)
            rng = np.random.default_rng(42)
            n_corrosion = int(roof_pixels * corrosion_percent / 100.0)
            roof_indices = np.argwhere(roof_mask)
            if len(roof_indices) > 0:
                chosen = rng.choice(len(roof_indices), size=min(n_corrosion, len(roof_indices)), replace=False)
                for idx in chosen:
                    r, c = roof_indices[idx]
                    corrosion_mask[r, c] = True

        # ── 4. Build result ──────────────────────────────────────
        severity = vlm_result["severity"]
        # If VLM severity disagrees with percent, trust percent
        derived = classify_severity(corrosion_percent)
        if severity != derived:
            logger.debug(f"Severity mismatch: VLM={severity}, derived={derived}; using derived")
            severity = derived

        confidence = float(vlm_result.get("confidence", 0.5))
        if not vlm_result.get("assessable", True):
            confidence = min(confidence, 0.3)

        return CorrosionResult(
            roof_area_m2=float(roof_area_m2),
            corroded_area_m2=float(corroded_area_m2),
            corrosion_percent=float(corrosion_percent),
            severity=severity,
            confidence=confidence,
            roof_mask=roof_mask,
            corrosion_mask=corrosion_mask,
            gsd=gsd,
            roof_model_version=f"{footprint_info.get('source', 'unknown')}:{footprint_info.get('osm_id', 'n/a')}",
            corrosion_model_version=self.nim.model,
        )

    # ── Roof-mask strategy ───────────────────────────────────

    async def _get_roof_mask(
        self,
        tile_image: np.ndarray,
        lat: Optional[float],
        lng: Optional[float],
        tile_bounds: Optional[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, dict]:
        """Produce a roof mask using OSM (preferred) then SAM (fallback).

        Returns: (mask, footprint_info)
        """
        h, w = tile_image.shape[:2]

        # Try OSM footprint first
        if self.config.use_osm_footprint and lat is not None and lng is not None:
            try:
                footprint = await self.footprint.get_footprint(
                    lat=lat, lng=lng, radius_m=self.config.footprint_radius_m
                )
            except Exception as e:
                logger.warning(f"Footprint query failed: {e}")
                footprint = None

            if footprint and tile_bounds and footprint["source"] == "osm":
                # Rasterize the OSM polygon into pixel space
                mask = self._rasterize_polygon(footprint["polygon_ll"], tile_bounds, (h, w))
                if mask.sum() > 50:  # sanity: at least 50 pixels
                    # Optionally refine with SAM
                    if self.config.sam_always and self._sam_available():
                        try:
                            # Use polygon centroid as a point prompt
                            cx, cy = self._mask_centroid(mask)
                            sam_mask = await self.sam.segment(tile_image, point_xy=(cx, cy))
                            # Intersect SAM mask with OSM polygon to avoid SAM grabbing neighbors
                            return (mask & sam_mask), footprint
                        except Exception as e:
                            logger.warning(f"SAM refinement failed: {e}; using OSM only")
                    return mask, footprint

        # Fallback: SAM with centered point prompt
        if self.config.use_sam_fallback and self._sam_available():
            try:
                center = (w // 2, h // 2)
                mask = await self.sam.segment(tile_image, point_xy=center)
                return mask, {"source": "sam", "osm_id": None, "area_m2": 0, "tags": {}}
            except Exception as e:
                logger.warning(f"SAM fallback failed: {e}; using full-tile mask")

        # Last resort: treat the full tile as roof
        return np.ones((h, w), dtype=bool), {"source": "full_tile", "osm_id": None, "area_m2": 0, "tags": {}}

    def _sam_available(self) -> bool:
        """Check whether SAM backend is usable."""
        try:
            backend = self.sam.backend
            return backend in ("runpod", "replicate") or backend == "local"
        except Exception:
            return False

    # ── Geometry utilities ───────────────────────────────────

    @staticmethod
    def _rasterize_polygon(
        polygon_ll: list[list[float]],
        tile_bounds: tuple[float, float, float, float],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Rasterize a lat/lng polygon into a pixel mask.

        Args:
            polygon_ll: [[lat, lng], ...]
            tile_bounds: (min_lat, min_lng, max_lat, max_lng)
            shape: (H, W) of output mask

        Returns:
            (H, W) bool numpy array
        """
        h, w = shape
        min_lat, min_lng, max_lat, max_lng = tile_bounds
        lat_range = max(max_lat - min_lat, 1e-9)
        lng_range = max(max_lng - min_lng, 1e-9)

        # Convert lat/lng to pixel coordinates
        # Note: in image coordinates, y increases downward = latitude decreases
        pixels = []
        for lat, lng in polygon_ll:
            x = int((lng - min_lng) / lng_range * w)
            y = int((max_lat - lat) / lat_range * h)
            pixels.append((x, y))

        # Use PIL for polygon rasterization (avoids cv2 dependency)
        from PIL import Image as _Image, ImageDraw
        mask_img = _Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(pixels, outline=1, fill=1)
        return np.array(mask_img).astype(bool)

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
        """Return (x, y) pixel centroid of a boolean mask."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return mask.shape[1] // 2, mask.shape[0] // 2
        return int(xs.mean()), int(ys.mean())

    @staticmethod
    def _crop_to_roof(
        tile_image: np.ndarray,
        roof_mask: np.ndarray,
        padding: int = 20,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the tile to the roof bounding box with padding.

        Returns:
            (cropped_image, (x1, y1, x2, y2))
        """
        h, w = tile_image.shape[:2]
        ys, xs = np.where(roof_mask)
        if len(xs) == 0:
            return tile_image, (0, 0, w, h)

        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(w, int(xs.max()) + padding)
        y2 = min(h, int(ys.max()) + padding)

        cropped = tile_image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)


# ═══════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════

def create_fm_pipeline(config: Optional[FMPipelineConfig] = None) -> FoundationModelPipeline:
    """Create a foundation-model pipeline with default clients."""
    return FoundationModelPipeline(config=config)
