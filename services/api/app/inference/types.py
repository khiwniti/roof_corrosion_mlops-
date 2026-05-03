"""Shared data types for the inference pipeline.

Extracted from pipeline.py so that the FM (Foundation Model) pipeline can
import CorrosionResult without pulling in torch (which isn't installed
in the lightweight inference container).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


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
    roof_model_version: str = ""
    corrosion_model_version: str = ""
    # Optional metadata from FM pipeline
    roof_material: str = "unknown"
    description: str = ""


def classify_severity(corrosion_percent: float) -> str:
    """Map corrosion percentage to severity label."""
    if corrosion_percent < 5:
        return "none"
    elif corrosion_percent < 25:
        return "light"
    elif corrosion_percent < 50:
        return "moderate"
    else:
        return "severe"
