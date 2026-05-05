"""RunPod Serverless handler — Tiered Hybrid pipeline (Phase 1a).

Input format (from Vercel Route Handler):
    {
        "input": {
            "polygon": {"type": "Polygon", "coordinates": [...]},
            "tier": 0,
            "jobId": "uuid",
            "webhook": "https://app.vercel.app/api/runpod/webhook"
        }
    }

Phase 1a behaviour:
- tier=0: S2 ingestion → cloud mask → composite → U-Net stub segmentation
- tier>=1: placeholder (returns status "needs_upgrade")

Architecture: ADR-001, ADR-003, ADR-004, ADR-005
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

# Ensure ml/ is on path for imports
_ml_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ml"))
sys.path.insert(0, _ml_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tiered_handler")

# ── R2 / S3 upload ─────────────────────────────────────────────────────────

def _get_s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_mask_to_r2(mask_bytes: bytes, key: str, content_type: str = "image/png") -> str:
    bucket = os.environ.get("R2_BUCKET_NAME", "roof-corrosion-masks")
    s3 = _get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=mask_bytes, ContentType=content_type)
    public_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    if public_url:
        return f"{public_url}/{key}"
    return f"{os.environ.get('R2_ENDPOINT_URL', '')}/{bucket}/{key}"


# ── Webhook caller ────────────────────────────────────────────────────────

def call_webhook(webhook_url: str, payload: dict) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            logger.info("Webhook response: %s", resp.status)
    except Exception as e:
        logger.warning("Webhook call failed: %s", e)


# ── Ingestion + inference pipeline (with graceful stub fallback) ──────────

def _try_ingest_and_classify(polygon: dict) -> tuple[np.ndarray, dict[str, Any]]:
    """Attempt real S2 ingestion → features → classifier; fall back to stub.

    Returns (material_mask_png_uint8, result_dict).
    """
    try:
        from ingestion.cdse import fetch_s2_single_composite
        from ingestion.cloud_mask import mask_stack
        from ingestion.composite import composite_with_fallback
        from ingestion.features import extract_features, normalize_features
        from inference.tier0 import predict, estimate_roof_area
        from ingestion.overture import fetch_buildings, compute_roof_area

        logger.info("Attempting real S2 ingestion via CDSE...")
        da = fetch_s2_single_composite(polygon, lookback_days=30, max_cloud=50, resolution=10)

        # Cloud mask
        da_masked = mask_stack(da, use_omnicloud=True, use_scl=True, use_s2cloudless=False)

        # Composite with fallback
        composite, meta = composite_with_fallback(da_masked, min_coverage=0.5)

        # Feature extraction
        features, feature_names = extract_features(composite, include_glcm=True, include_s1=False)
        features = normalize_features(features, method="minmax")

        # Classifier
        pred = predict(features, feature_names)
        material_mask = pred["material_mask"]

        # Building footprints from Overture
        try:
            gdf = fetch_buildings(polygon)
            bldg_meta = compute_roof_area(gdf)
            building_count = bldg_meta["building_count"]
        except Exception as e:
            logger.warning("Overture fetch failed: %s", e)
            building_count = None

        # Area estimation
        area_meta = estimate_roof_area(material_mask, gsd_m=10.0, building_count=building_count)

        # Build colored mask PNG for visualization (class -> color)
        colors = np.array([
            [180, 180, 180],  # metal — grey
            [220, 120, 60],   # tile — terracotta
            [160, 160, 160],  # concrete — light grey
            [34, 139, 34],    # vegetation — green
            [128, 128, 128],  # other — mid grey
        ], dtype=np.uint8)
        colored_mask = colors[material_mask]

        meta["ingestion_method"] = "cdse_real"
        return colored_mask, {
            "ingestion_meta": meta,
            "prediction": pred,
            "area": area_meta,
            "building_count": building_count,
        }

    except Exception as e:
        logger.warning("Real pipeline failed (%s); using stub", e)
        # Fallback: generate a random material mask
        h, w = 512, 512
        np.random.seed(42)
        mask = np.random.randint(0, 3, size=(h, w), dtype=np.uint8)
        colors = np.array([
            [180, 180, 180], [220, 120, 60], [160, 160, 160],
            [34, 139, 34], [128, 128, 128],
        ], dtype=np.uint8)
        colored_mask = colors[mask]
        return colored_mask, {
            "ingestion_meta": {"ingestion_method": "stub", "error": str(e)},
            "prediction": {
                "material_mask": mask,
                "class_names": ["metal", "tile", "concrete", "vegetation", "other"],
                "class_areas": {},
                "class_percentages": {},
                "coarse_breakdown": {"metal_percent": 30.0, "tile_percent": 40.0, "other_percent": 30.0},
                "confidence": 0.5,
                "model_version": "tier0-stub-v0",
            },
            "area": {"roof_area_m2": 2500.0, "building_count": 0, "avg_area_per_building_m2": None, "pixel_area_m2": 100.0},
            "building_count": 0,
        }


# ── Main handler ────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    t0 = time.time()
    inp = job.get("input", {})

    polygon = inp.get("polygon")
    tier = int(inp.get("tier", 0))
    job_id = inp.get("jobId", "")
    # RunPod passes webhook at top level; fallback to input field for local testing
    webhook = job.get("webhook", "") or inp.get("webhook", "")

    if not polygon or not job_id:
        return {"error": "Missing polygon or jobId", "status": "failed"}

    logger.info("Job %s tier=%s started", job_id, tier)

    # Phase 1a: Tier 0 full pipeline
    if tier == 0:
        colored_mask, result = _try_ingest_and_classify(polygon)
        pred = result["prediction"]
        area = result["area"]
        ingest_meta = result["ingestion_meta"]

        img = Image.fromarray(colored_mask, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mask_bytes = buf.getvalue()

        key = f"masks/{job_id}_tier0.png"
        try:
            mask_url = upload_mask_to_r2(mask_bytes, key)
        except Exception as e:
            logger.warning("R2 upload failed (%s), returning base64", e)
            import base64
            mask_url = f"data:image/png;base64,{base64.b64encode(mask_bytes).decode()}"

        # Call Vercel webhook with Tier-0 results
        if webhook:
            call_webhook(webhook, {
                "jobId": job_id,
                "status": "completed",
                "mask_url": mask_url,
                "model_version": pred["model_version"],
                "confidence": pred["confidence"],
                "area_m2": area["roof_area_m2"],
                "building_count": area["building_count"],
                "class_areas": pred.get("class_areas", {}),
                "class_percentages": pred.get("class_percentages", {}),
                "coarse_breakdown": pred.get("coarse_breakdown", {}),
                "ingestion_meta": ingest_meta,
            })

        # Clean non-JSON-serializable ndarrays from prediction before returning
        prediction_summary = {k: v for k, v in pred.items() if not isinstance(v, np.ndarray)}

        processing_ms = int((time.time() - t0) * 1000)
        return {
            "status": "completed",
            "jobId": job_id,
            "mask_url": mask_url,
            "processing_ms": processing_ms,
            "area": area,
            "prediction": prediction_summary,
            "ingestion_meta": ingest_meta,
        }

    # Tier 1+ placeholder
    if webhook:
        call_webhook(webhook, {
            "jobId": job_id,
            "status": "needs_upgrade",
            "message": f"Tier {tier} requires paid imagery. Please upgrade.",
        })

    return {
        "status": "needs_upgrade",
        "jobId": job_id,
        "message": f"Tier {tier} not yet implemented. Use Tier 0 for free preliminary.",
    }


if __name__ == "__main__":
    test_job = {
        "input": {
            "polygon": {"type": "Polygon", "coordinates": [[[100.5, 13.7], [100.51, 13.7], [100.51, 13.71], [100.5, 13.71], [100.5, 13.7]]]},
            "tier": 0,
            "jobId": "test-job-001",
            "webhook": "",
        }
    }
    result = handler(test_job)
    print(json.dumps(result, indent=2))
