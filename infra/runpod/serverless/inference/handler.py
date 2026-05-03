"""RunPod Serverless inference handler — Foundation Model pipeline.

Zero-GPU corrosion detection using NVIDIA NIM VLM + OSM Building Footprints.
No local models to load — all heavy inference runs via NVIDIA's API.

Cold-start time: ~2s (Python + httpx imports only)
Per-request cost: ~$0.01–0.05 (NIM VLM tokens, depending on image size)

Deploy:
    docker build --platform linux/amd64 -t roof-corrosion-inference .
    docker push YOUR_REGISTRY/roof-corrosion-inference:latest

Local test:
    python handler.py --test_input test_input.json

API request format:
    POST /runsync
    {
        "input": {
            "image_url": "https://...",        # satellite tile URL
            "image_base64": "data:image/png;base64,...",  # or base64
            "lat": 13.7563,                    # WGS84 lat (enables OSM lookup)
            "lng": 100.5018,                   # WGS84 lng
            "gsd": 0.3,                        # ground sample distance m/pixel
            "address": "123 Sukhumvit Rd",    # optional: NIM prompt context
            "material": "corrugated_metal",    # optional: roof material
            "region": "TH",                    # optional: pricing region
            "return_masks": false              # include base64 mask PNGs in response
        }
    }
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import runpod
from PIL import Image

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("handler")

# ── Import app package ───────────────────────────────────────────────────────
# The Dockerfile copies services/api/app/ → /app/src/app/
# and sets PYTHONPATH=/app/src so we can import from app.*
try:
    from app.inference.pipeline_fm import FoundationModelPipeline, FMPipelineConfig
    from app.quote_engine import compute_quote
    from app.region import get_active_region
    _APP_AVAILABLE = True
    logger.info("app package loaded OK")
except ImportError as _e:
    logger.warning(f"app package not importable ({_e}); running in stub mode")
    _APP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Image loading
# ═══════════════════════════════════════════════════════════════════════════

def load_image_from_url(url: str) -> np.ndarray:
    """Download an image from a URL and return (H, W, 3) uint8 RGB array."""
    import urllib.request
    logger.info(f"Fetching image from URL: {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def load_image_from_base64(b64: str) -> np.ndarray:
    """Decode a base64 string (with or without data URI prefix)."""
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def mask_to_b64(mask: np.ndarray) -> str:
    """Convert a bool mask to a base64-encoded PNG string."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline (module-level singleton — stays warm between requests)
# ═══════════════════════════════════════════════════════════════════════════

_pipeline: Optional["FoundationModelPipeline"] = None


def get_pipeline() -> "FoundationModelPipeline":
    global _pipeline
    if _pipeline is None:
        if not _APP_AVAILABLE:
            raise RuntimeError("app package unavailable — check container build")
        config = FMPipelineConfig(
            use_osm_footprint=True,
            use_sam_fallback=True,
            min_confidence=float(os.environ.get("MIN_CONFIDENCE", "0.6")),
        )
        _pipeline = FoundationModelPipeline(config=config)
        logger.info("FoundationModelPipeline initialised (NIM + OSM)")
    return _pipeline


# ═══════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _validate_input(inp: dict) -> str | None:
    """Return an error message string if the input is invalid, else None."""
    has_image = bool(inp.get("image_url") or inp.get("image_base64"))
    has_coords = inp.get("lat") is not None and inp.get("lng") is not None
    if not has_image and not has_coords:
        return "Provide at least one of: image_url, image_base64, or lat+lng"
    gsd = inp.get("gsd", 0.3)
    if not (0.05 <= float(gsd) <= 10.0):
        return f"gsd must be between 0.05 and 10.0 m/pixel, got {gsd}"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Tile fetching when only lat/lng is provided
# ═══════════════════════════════════════════════════════════════════════════

async def _fetch_tile_for_coords(lat: float, lng: float) -> tuple[np.ndarray, float, tuple]:
    """Fetch an Esri World Imagery tile for the given coordinates.

    Returns (tile_array, gsd_m, tile_bounds).
    Falls back to a blank gray tile if the fetch fails.
    """
    try:
        from app.inference.tile_fetch import TileFetcher
        fetcher = TileFetcher()
        job = {"lat": lat, "lng": lng, "source": None}
        # fetch_tiles_for_job is defined in worker, but we can call TileFetcher directly
        tile, gsd, bounds = await fetcher.fetch(lat=lat, lng=lng, zoom=19)
        return tile, gsd, bounds
    except Exception as e:
        logger.warning(f"Tile fetch failed ({e}), using blank 256×256 tile")
        blank = np.full((256, 256, 3), 128, dtype=np.uint8)
        return blank, 0.30, (lat - 0.001, lng - 0.001, lat + 0.001, lng + 0.001)


# ═══════════════════════════════════════════════════════════════════════════
# Main async handler
# ═══════════════════════════════════════════════════════════════════════════

async def _run(job: dict) -> dict:
    """Core async logic — separated so we can call it from sync handler."""
    inp = job.get("input", {})
    t0 = time.time()

    # ── 0. Validate ──────────────────────────────────────────────────────────
    err = _validate_input(inp)
    if err:
        return {"error": err, "status": "failed"}

    # ── 1. Load satellite image ───────────────────────────────────────────────
    lat: Optional[float] = inp.get("lat")
    lng: Optional[float] = inp.get("lng")
    gsd: float = float(inp.get("gsd", 0.30))
    tile_bounds: Optional[tuple] = None

    if inp.get("image_url"):
        try:
            tile_image = load_image_from_url(inp["image_url"])
        except Exception as e:
            return {"error": f"Failed to fetch image_url: {e}", "status": "failed"}

    elif inp.get("image_base64"):
        try:
            tile_image = load_image_from_base64(inp["image_base64"])
        except Exception as e:
            return {"error": f"Failed to decode image_base64: {e}", "status": "failed"}

    else:
        # No image provided — fetch a satellite tile for the coordinates
        tile_image, gsd, tile_bounds = await _fetch_tile_for_coords(lat, lng)

    logger.info(f"Tile loaded: shape={tile_image.shape} gsd={gsd:.3f}m/px")

    # ── 2. Run FM pipeline ────────────────────────────────────────────────────
    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        return {"error": str(e), "status": "failed"}

    try:
        result = await pipeline.analyze(
            tile_image=tile_image,
            lat=lat,
            lng=lng,
            gsd=gsd,
            address=inp.get("address", ""),
            tile_bounds=tile_bounds,
        )
    except Exception as e:
        logger.exception("FM pipeline failed")
        return {"error": f"Inference failed: {e}", "status": "failed"}

    # ── 3. Compute quote ──────────────────────────────────────────────────────
    region_code = inp.get("region", os.environ.get("REGION", "TH"))
    material = inp.get("material", "corrugated_metal")
    try:
        os.environ["REGION"] = region_code  # region.get_active_region() reads env
        quote = compute_quote(
            roof_area_m2=result.roof_area_m2,
            corroded_area_m2=result.corroded_area_m2,
            corrosion_percent=result.corrosion_percent,
            severity=result.severity,
            confidence=result.confidence,
            material=material,
            region=region_code,
        )
    except Exception as e:
        logger.warning(f"Quote computation failed ({e}); returning assessment only")
        quote = None

    # ── 4. Build response ─────────────────────────────────────────────────────
    processing_ms = int((time.time() - t0) * 1000)

    response: dict = {
        "status": "completed",
        "processing_ms": processing_ms,
        "assessment": {
            "roof_area_m2": round(result.roof_area_m2, 2),
            "corroded_area_m2": round(result.corroded_area_m2, 2),
            "corrosion_percent": round(result.corrosion_percent, 1),
            "severity": result.severity,
            "confidence": round(result.confidence, 3),
            "roof_material": getattr(result, "roof_material", "unknown"),
            "description": getattr(result, "description", ""),
        },
        "model_versions": {
            "roof": result.roof_model_version,
            "corrosion": result.corrosion_model_version,
        },
        "gsd_m": gsd,
    }

    if quote is not None:
        response["quote"] = {
            "currency": quote.currency,
            "total_amount": quote.total_amount,
            "line_items": quote.line_items,
            "requires_human_review": quote.requires_human_review,
            "review_reason": quote.review_reason,
        }

    # Optional: include base64-encoded mask PNGs
    if inp.get("return_masks", False):
        response["masks"] = {
            "roof_png_b64": mask_to_b64(result.roof_mask),
            "corrosion_png_b64": mask_to_b64(result.corrosion_mask),
        }

    logger.info(
        f"Done: severity={result.severity} corrosion={result.corrosion_percent:.1f}% "
        f"area={result.roof_area_m2:.0f}m² time={processing_ms}ms"
    )
    return response


def handler(job: dict) -> dict:
    """RunPod Serverless entry point (sync wrapper around async core)."""
    try:
        return asyncio.run(_run(job))
    except Exception as e:
        logger.exception("Unhandled error in handler")
        return {"error": str(e), "status": "failed"}


# ═══════════════════════════════════════════════════════════════════════════
# Local test entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    test_input_path = Path(__file__).parent / "test_input.json"
    if "--test_input" in sys.argv:
        idx = sys.argv.index("--test_input")
        test_input_path = Path(sys.argv[idx + 1])

    with open(test_input_path) as f:
        job = json.load(f)

    print("Running local test with:", json.dumps(job, indent=2))
    result = handler(job)
    print("\nResult:", json.dumps(result, indent=2, default=str))
