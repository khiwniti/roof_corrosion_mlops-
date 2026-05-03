"""Two-stage inference pipeline: roof footprint → corrosion segmentation.

Stage 1: Mask2Former detects roof polygon
Stage 2: SegFormer/UNet++ segments corrosion within the roof crop
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrosionResult:
    """Output of the two-stage corrosion analysis pipeline."""

    roof_area_m2: float
    corroded_area_m2: float
    corrosion_percent: float
    severity: str  # none | light | moderate | severe
    confidence: float
    roof_mask: np.ndarray  # (H, W) binary
    corrosion_mask: np.ndarray  # (H, W) binary
    gsd: float  # meters per pixel


SEVERITY_THRESHOLDS = {
    "none": 0.0,
    "light": 5.0,
    "moderate": 25.0,
    "severe": 50.0,
}


def classify_severity(corrosion_percent: float) -> str:
    """Map corrosion percentage to severity grade."""
    for severity, threshold in reversed(SEVERITY_THRESHOLDS.items()):
        if corrosion_percent >= threshold:
            return severity
    return "none"


class CorrosionPipeline:
    """Two-stage roof corrosion analysis pipeline.

    Usage:
        pipeline = CorrosionPipeline(
            roof_model_uri="mlflow:///models/stage1_roof/production",
            corrosion_model_uri="mlflow:///models/stage2_corrosion/production",
        )
        result = pipeline.analyze(tile_image, gsd=0.3)
    """

    def __init__(
        self,
        roof_model_uri: str,
        corrosion_model_uri: str,
        device: str = "cuda",
    ):
        self.device = device
        # TODO: load models from MLflow
        self.roof_model = None
        self.corrosion_model = None

    def analyze(self, tile_image: np.ndarray, gsd: float = 0.3) -> CorrosionResult:
        """Run two-stage analysis on a satellite tile.

        Args:
            tile_image: (H, W, 3) uint8 RGB image
            gsd: ground sample distance in meters per pixel
        """
        # Stage 1: Roof footprint
        roof_mask = self._predict_roof(tile_image)

        # Stage 2: Corrosion on roof crop (with 10m buffer)
        buffer_pixels = int(10.0 / gsd)  # 10m buffer around roof
        corrosion_mask = self._predict_corrosion(tile_image, roof_mask, buffer_pixels)

        # Compute areas
        roof_pixels = roof_mask.sum()
        corrosion_pixels = (corrosion_mask & roof_mask).sum()
        roof_area_m2 = roof_pixels * gsd * gsd
        corroded_area_m2 = corrosion_pixels * gsd * gsd
        corrosion_percent = (corroded_area_m2 / roof_area_m2 * 100) if roof_area_m2 > 0 else 0.0

        severity = classify_severity(corrosion_percent)

        # TODO: compute confidence from MC-dropout or ensemble disagreement
        confidence = 0.5

        return CorrosionResult(
            roof_area_m2=roof_area_m2,
            corroded_area_m2=corroded_area_m2,
            corrosion_percent=corrosion_percent,
            severity=severity,
            confidence=confidence,
            roof_mask=roof_mask,
            corrosion_mask=corrosion_mask,
            gsd=gsd,
        )

    def _predict_roof(self, tile_image: np.ndarray) -> np.ndarray:
        """Stage 1: Predict roof footprint mask."""
        # TODO: implement with Mask2Former
        return np.zeros(tile_image.shape[:2], dtype=bool)

    def _predict_corrosion(
        self, tile_image: np.ndarray, roof_mask: np.ndarray, buffer_px: int
    ) -> np.ndarray:
        """Stage 2: Predict corrosion mask within roof crop + buffer."""
        # TODO: implement with SegFormer/UNet++
        return np.zeros(tile_image.shape[:2], dtype=bool)
