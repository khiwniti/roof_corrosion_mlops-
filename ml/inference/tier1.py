"""Tier-1 inference: VHR imagery → detector → SAM2 → classifier → calibrated output.

Architecture Decision: ADR-001, ADR-008, ADR-013
- VHR ingestion: Pléiades or THEOS-2
- Building detection: YOLO or Mask R-CNN
- Roof segmentation: SAM 2.1 box-prompted
- Classification: Clay v1.5 multi-task model (from Phase 2)
- Calibration: temperature scaling per province

This is a Phase 3 stub. Production integrates real SAM 2.1 and detector weights.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("inference.tier1")


def detect_buildings(
    vhr_image: np.ndarray,
    confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect buildings in VHR image using YOLO / Mask R-CNN.

    Parameters
    ----------
    vhr_image : (H, W, 3) uint8 RGB image
    confidence : detection confidence threshold

    Returns
    -------
    list of dicts with keys: bbox [x1,y1,x2,y2], score, class_id
    """
    logger.info("Building detection stub on %s image", vhr_image.shape)
    # Stub: return a single centered bbox
    h, w = vhr_image.shape[:2]
    return [
        {
            "bbox": [w * 0.2, h * 0.2, w * 0.8, h * 0.8],
            "score": 0.85,
            "class_id": 0,  # building
        }
    ]


def segment_roof_sam2(
    vhr_image: np.ndarray,
    bbox: list[float],
) -> np.ndarray:
    """Segment roof mask from VHR image using SAM 2.1 box prompt.

    Parameters
    ----------
    vhr_image : (H, W, 3) uint8 RGB
    bbox : [x1, y1, x2, y2]

    Returns
    -------
    mask : (H, W) bool array
    """
    logger.info("SAM2 segmentation stub for bbox %s", bbox)
    h, w = vhr_image.shape[:2]
    # Stub: elliptical mask inside bbox
    mask = np.zeros((h, w), dtype=bool)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    rx, ry = (bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2
    yv, xv = np.ogrid[:h, :w]
    ellipse = ((xv - cx) / rx) ** 2 + ((yv - cy) / ry) ** 2 <= 1
    mask[ellipse] = True
    return mask


def classify_roof(
    roof_crop: np.ndarray,
    model: Any,
) -> dict[str, Any]:
    """Run Clay multi-task classifier on a single roof crop.

    Parameters
    ----------
    roof_crop : (H, W, 3) uint8 RGB or (H, W, 10) S2 stack
    model : ClayMultiTaskModel or stub

    Returns
    -------
    dict with material_probs, corrosion_prob, severity_probs
    """
    logger.info("Classification stub on crop %s", roof_crop.shape)
    # Stub: return plausible random probabilities
    return {
        "material_probs": [0.6, 0.2, 0.1, 0.05, 0.05],  # metal, tile, concrete, veg, other
        "corrosion_prob": 0.35,
        "severity_probs": [0.5, 0.3, 0.15, 0.05],  # none, light, moderate, severe
        "confidence": 0.72,
    }


def temperature_scaling(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply temperature scaling to calibrate confidence probabilities.

    Parameters
    ----------
    logits : (N, C) raw model logits
    temperature : learned scalar per province (default 1.0 = no calibration)

    Returns
    -------
    calibrated probabilities
    """
    return F.softmax(logits / temperature, dim=-1)


def predict(
    vhr_image: np.ndarray,
    model: Any,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Full Tier-1 inference pipeline.

    1. Detect buildings
    2. Segment each roof with SAM2
    3. Classify each roof crop
    4. Calibrate probabilities with temperature scaling
    5. Aggregate area and material breakdown
    """
    buildings = detect_buildings(vhr_image)
    results = []
    total_area_px = 0

    for bldg in buildings:
        mask = segment_roof_sam2(vhr_image, bldg["bbox"])
        area_px = int(mask.sum())
        total_area_px += area_px

        # Crop to bbox for classifier
        x1, y1, x2, y2 = map(int, bldg["bbox"])
        crop = vhr_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls = classify_roof(crop, model)

        # Apply temperature scaling to material logits (stub)
        mat_logits = torch.tensor(cls["material_probs"]).unsqueeze(0)
        mat_calibrated = temperature_scaling(mat_logits, temperature)
        cls["material_probs_calibrated"] = mat_calibrated.squeeze(0).tolist()

        results.append({
            "bbox": bldg["bbox"],
            "area_px": area_px,
            **cls,
        })

    # Aggregate
    if results:
        material_counts = np.zeros(5)
        for r in results:
            material_counts[np.argmax(r["material_probs"])] += r["area_px"]
        total = material_counts.sum()
        material_pct = (material_counts / total * 100).tolist() if total > 0 else [0] * 5
    else:
        material_pct = [0, 0, 0, 0, 100]

    return {
        "buildings": results,
        "building_count": len(results),
        "total_roof_area_px": total_area_px,
        "material_breakdown": {
            "metal_percent": round(material_pct[0], 1),
            "tile_percent": round(material_pct[1], 1),
            "concrete_percent": round(material_pct[2], 1),
            "vegetation_percent": round(material_pct[3], 1),
            "other_percent": round(material_pct[4], 1),
        },
        "corrosion_detected": any(r["corrosion_prob"] > 0.3 for r in results),
        "temperature": temperature,
        "model_version": "tier1-stub-v0",
        "confidence": 0.72,
    }
