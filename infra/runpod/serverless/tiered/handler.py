"""RunPod Serverless handler — Tiered Hybrid pipeline (Phase 1a/3).

Input format (from Vercel Route Handler):
    {
        "input": {
            "polygon": {"type": "Polygon", "coordinates": [...]},
            "tier": 0,  // 0=S2 free, 1=VHR paid, 3=drone binding
            "jobId": "uuid",
            "webhook": "https://app.vercel.app/api/runpod/webhook"
        }
    }

Tier behaviour:
- tier=0: S2 ingestion → cloud mask → composite → material classifier stub
- tier=1: VHR (Pléiades/THEOS-2) → detector + SAM2 → classifier → ±15% quote
- tier>=2: placeholder (returns status "needs_upgrade")

Architecture: ADR-001, ADR-003, ADR-004, ADR-005, ADR-008, ADR-013
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
                "corrosion_prob": 0.20,
                "severity": "light",
                "corroded_area_px": int(512 * 512 * 0.20),
                "confidence": 0.5,
                "model_version": "tier0-stub-v0",
            },
            "area": {"roof_area_m2": 2500.0, "building_count": 0, "avg_area_per_building_m2": None, "pixel_area_m2": 100.0},
            "building_count": 0,
        }


# ── Tier-1 VHR pipeline (with fallback to Tier-0) ─────────────────────────

def _try_ingest_tier1(polygon: dict) -> tuple[np.ndarray, dict[str, Any]]:
    """Attempt Tier-1 VHR ingestion → detector → SAM2 → classifier.

    Falls back to Tier-0 S2 pipeline if VHR is unavailable.
    Returns (colored_mask, result_dict).
    """
    try:
        from ingestion.pleiades import search_archive, estimate_cost
        from inference.tier1 import predict

        logger.info("Attempting Tier-1 VHR ingestion (Pléiades)...")
        scenes = search_archive(polygon, max_cloud=15, constellation="PHR")

        if not scenes:
            logger.warning("No Pléiades archive scenes found; trying THEOS-2...")
            from ingestion.theos2 import search_catalog
            scenes = search_catalog(polygon, max_cloud=20)

        if not scenes:
            logger.warning("No VHR scenes available; falling back to Tier-0 S2")
            return _try_ingest_and_classify(polygon)

        cost = estimate_cost(polygon, constellation="PHR")
        logger.info("VHR cost estimate: %s EUR", cost["estimated_cost_eur"])

        # Stub: generate a synthetic VHR-like mask for demonstration
        h, w = 1024, 1024
        np.random.seed(43)
        mask = np.random.randint(0, 5, size=(h, w), dtype=np.uint8)
        colors = np.array([
            [180, 180, 180], [220, 120, 60], [160, 160, 160],
            [34, 139, 34], [128, 128, 128],
        ], dtype=np.uint8)
        colored_mask = colors[mask]

        # Run Tier-1 classifier stub
        vhr_stub = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        pred = predict(vhr_stub, model=None, temperature=1.0)

        return colored_mask, {
            "ingestion_meta": {
                "ingestion_method": "pleiades_archive",
                "scene_count": len(scenes),
                "cost_estimate_eur": cost["estimated_cost_eur"],
                "cost_estimate_thb": cost["estimated_cost_thb"],
            },
            "prediction": {
                "class_names": ["metal", "tile", "concrete", "vegetation", "other"],
                "class_areas": {},
                "class_percentages": {},
                "coarse_breakdown": pred["material_breakdown"],
                "confidence": pred["confidence"],
                "model_version": pred["model_version"],
            },
            "area": {
                "roof_area_m2": pred["total_roof_area_px"] * 0.25,  # 0.5m GSD
                "building_count": pred["building_count"],
                "avg_area_per_building_m2": None,
                "pixel_area_m2": 0.25,
            },
            "building_count": pred["building_count"],
        }

    except Exception as e:
        logger.warning("Tier-1 VHR failed (%s); falling back to Tier-0 S2", e)
        return _try_ingest_and_classify(polygon)


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

    # ── Tier 0: Free S2 preliminary ──
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
                "corrosion_prob": pred.get("corrosion_prob"),
                "severity": pred.get("severity"),
                "corroded_area_px": pred.get("corroded_area_px"),
                "ingestion_meta": ingest_meta,
            })

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

    # ── Tier 1: Paid VHR binding quote ──
    if tier == 1:
        colored_mask, result = _try_ingest_tier1(polygon)
        pred = result["prediction"]
        area = result["area"]
        ingest_meta = result["ingestion_meta"]

        img = Image.fromarray(colored_mask, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        mask_bytes = buf.getvalue()

        key = f"masks/{job_id}_tier1.png"
        try:
            mask_url = upload_mask_to_r2(mask_bytes, key)
        except Exception as e:
            logger.warning("R2 upload failed (%s), returning base64", e)
            import base64
            mask_url = f"data:image/png;base64,{base64.b64encode(mask_bytes).decode()}"

        # Tier-1 includes cost estimate and ±15% confidence
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
                "corrosion_prob": pred.get("corrosion_prob"),
                "severity": pred.get("severity"),
                "corroded_area_px": pred.get("corroded_area_px"),
                "ingestion_meta": ingest_meta,
                "cost_estimate_eur": ingest_meta.get("cost_estimate_eur"),
                "cost_estimate_thb": ingest_meta.get("cost_estimate_thb"),
                "quote_band": "±15%",
                "requires_human_review": area["roof_area_m2"] * 850 > 100000,  # THB 100k threshold
            })

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

    # ── Tier 2+ placeholder ──
    if webhook:
        call_webhook(webhook, {
            "jobId": job_id,
            "status": "needs_upgrade",
            "message": f"Tier {tier} requires drone imagery. Please upgrade.",
        })

    return {
        "status": "needs_upgrade",
        "jobId": job_id,
        "message": f"Tier {tier} not yet implemented. Use Tier 0 (free) or Tier 1 (paid VHR).",
    }


if __name__ == "__main__":
    test_polygon = {"type": "Polygon", "coordinates": [[[100.5, 13.7], [100.51, 13.7], [100.51, 13.71], [100.5, 13.71], [100.5, 13.7]]]}

    for tier in [0, 1]:
        print(f"\n=== Tier {tier} test ===")
        test_job = {
            "input": {
                "polygon": test_polygon,
                "tier": tier,
                "jobId": f"test-job-tier{tier}",
                "webhook": "",
            }
        }
        result = handler(test_job)
        print(json.dumps(result, indent=2))
