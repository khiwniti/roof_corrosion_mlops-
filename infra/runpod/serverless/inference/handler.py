"""RunPod Serverless inference handler for roof corrosion detection.

Two-stage pipeline: roof footprint (SegFormer-B3) → corrosion mask (SegFormer-B2).
Models loaded from TorchScript files on network volume or baked into image.

Deploy:
    docker build -t roof-corrosion-inference .
    docker push YOUR_REGISTRY/roof-corrosion-inference:latest

Local test:
    python handler.py --test_input test_input.json

API request format:
    POST /runsync
    {
        "input": {
            "image_url": "https://...",     # URL or base64-encoded image
            "image_base64": "data:image/png;base64,...",  # alternative
            "gsd": 0.3,                     # ground sample distance (m/pixel)
            "address": "123 Main St",        # optional, for quote generation
            "return_masks": false            # return mask arrays (default: false)
        }
    }
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import runpod
import torch
import torch.nn.functional as F
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model references (loaded once at cold start)
roof_model = None
corrosion_model = None
model_versions = {"roof": "not_loaded", "corrosion": "not_loaded"}


def load_models():
    """Load TorchScript models from disk. Called once at worker startup."""
    global roof_model, corrosion_model, model_versions

    roof_path = MODELS_DIR / "stage1_roof_footprint.pt"
    corrosion_path = MODELS_DIR / "stage2_corrosion.pt"

    # Try TorchScript first, then HuggingFace fallback
    if roof_path.exists():
        roof_model = torch.jit.load(str(roof_path), map_location=DEVICE)
        roof_model.eval()
        model_versions["roof"] = f"torchscript:{roof_path.name}"
        print(f"Loaded roof model from {roof_path}")
    else:
        print(f"No TorchScript roof model at {roof_path}, loading HuggingFace...")
        try:
            from transformers import SegformerForSemanticSegmentation
            hf_id = "nvidia/segformer-b3-finetuned-ade-512-512"
            base = SegformerForSemanticSegmentation.from_pretrained(
                hf_id, num_labels=2, ignore_mismatched_sizes=True
            )
            roof_model = base.to(DEVICE)
            roof_model.eval()
            model_versions["roof"] = f"huggingface:{hf_id}"
            print(f"Loaded roof model from HuggingFace: {hf_id}")
        except Exception as e:
            print(f"WARNING: Could not load roof model: {e}")
            roof_model = None

    if corrosion_path.exists():
        corrosion_model = torch.jit.load(str(corrosion_path), map_location=DEVICE)
        corrosion_model.eval()
        model_versions["corrosion"] = f"torchscript:{corrosion_path.name}"
        print(f"Loaded corrosion model from {corrosion_path}")
    else:
        print(f"No TorchScript corrosion model at {corrosion_path}, loading HuggingFace...")
        try:
            from transformers import SegformerForSemanticSegmentation
            hf_id = "nvidia/segformer-b2-finetuned-ade-512-512"
            base = SegformerForSemanticSegmentation.from_pretrained(
                hf_id, num_labels=3, ignore_mismatched_sizes=True
            )
            corrosion_model = base.to(DEVICE)
            corrosion_model.eval()
            model_versions["corrosion"] = f"huggingface:{hf_id}"
            print(f"Loaded corrosion model from HuggingFace: {hf_id}")
        except Exception as e:
            print(f"WARNING: Could not load corrosion model: {e}")
            corrosion_model = None

    print(f"Models loaded on {DEVICE}. Roof: {model_versions['roof']}, Corrosion: {model_versions['corrosion']}")


# ═══════════════════════════════════════════════════════════════
# Image utilities
# ═══════════════════════════════════════════════════════════════

def load_image(image_input: str) -> np.ndarray:
    """Load image from URL or base64 string.

    Returns:
        (H, W, 3) uint8 RGB numpy array
    """
    if image_input.startswith("http://") or image_input.startswith("https://"):
        import urllib.request
        resp = urllib.request.urlopen(image_input, timeout=30)
        img_bytes = resp.read()
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    # Base64
    if image_input.startswith("data:image"):
        # Strip data URI prefix
        b64_data = image_input.split(",", 1)[1]
    else:
        b64_data = image_input

    img_bytes = base64.b64decode(b64_data)
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def preprocess(tile_image: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) uint8 to (1, 3, H, W) normalized tensor."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(tile_image)
    return transform(pil).unsqueeze(0).to(DEVICE)


# ═══════════════════════════════════════════════════════════════
# Inference pipeline
# ═══════════════════════════════════════════════════════════════

SEVERITY_THRESHOLDS = {
    "none": 0.0,
    "light": 5.0,
    "moderate": 25.0,
    "severe": 50.0,
}


def classify_severity(corrosion_percent: float) -> str:
    for severity, threshold in reversed(SEVERITY_THRESHOLDS.items()):
        if corrosion_percent >= threshold:
            return severity
    return "none"


def predict_roof(img_tensor: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """Stage 1: Predict roof footprint mask."""
    if roof_model is None:
        return np.ones(original_shape, dtype=bool)  # fallback: assume whole image is roof

    with torch.no_grad():
        if hasattr(roof_model, 'predict_mask'):
            # SegFormer wrapper
            pred = roof_model.predict_mask(img_tensor)  # (1, H, W)
        else:
            # TorchScript or raw model
            output = roof_model(img_tensor)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("out", None))
            else:
                logits = output

            if logits.shape[-2:] != img_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)  # (1, H, W)

    mask = pred.cpu().numpy()[0]  # (H, W)
    return (mask > 0).astype(bool)


def predict_corrosion(img_tensor: torch.Tensor, roof_mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Stage 2: Predict corrosion mask within roof area."""
    if corrosion_model is None:
        return np.zeros(original_shape, dtype=bool)

    with torch.no_grad():
        if hasattr(corrosion_model, 'predict_mask'):
            pred = corrosion_model.predict_mask(img_tensor)
        else:
            output = corrosion_model(img_tensor)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("out", None))
            else:
                logits = output

            if logits.shape[-2:] != img_tensor.shape[-2:]:
                logits = F.interpolate(logits, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)

    mask = pred.cpu().numpy()[0]
    corrosion = (mask == 2).astype(bool)  # class 2 = corrosion
    return corrosion & roof_mask  # only within roof


def compute_confidence(img_tensor: torch.Tensor, corrosion_mask: np.ndarray) -> float:
    """Compute model confidence score.

    For now: ratio of high-confidence corrosion pixels.
    Production: MC-dropout uncertainty estimation.
    """
    if corrosion_model is None:
        return 0.5

    with torch.no_grad():
        output = corrosion_model(img_tensor)
        if isinstance(output, dict):
            logits = output.get("logits", output.get("out", None))
        else:
            logits = output

        if logits.shape[-2:] != img_tensor.shape[-2:]:
            logits = F.interpolate(logits, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1)
        corrosion_probs = probs[0, 2].cpu().numpy()  # (H, W)

    # Confidence = mean probability on predicted corrosion pixels
    if corrosion_mask.sum() > 0:
        mean_prob = corrosion_probs[corrosion_mask].mean()
    else:
        mean_prob = 1.0 - corrosion_probs.max()  # confidence in "no corrosion"

    return float(np.clip(mean_prob, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# Handler
# ═══════════════════════════════════════════════════════════════

def handler(job: dict) -> dict:
    """RunPod Serverless handler for roof corrosion inference.

    Args:
        job: {"id": "...", "input": {"image_url": "...", "gsd": 0.3, ...}}

    Returns:
        {
            "roof_area_m2": float,
            "corroded_area_m2": float,
            "corrosion_percent": float,
            "severity": str,
            "confidence": float,
            "roof_mask_b64": str | None,  # base64 PNG if return_masks=true
            "corrosion_mask_b64": str | None,
            "model_versions": dict,
            "inference_time_ms": float,
        }
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    # Extract image
    image_url = job_input.get("image_url", "")
    image_base64 = job_input.get("image_base64", "")
    gsd = job_input.get("gsd", 0.3)
    return_masks = job_input.get("return_masks", False)

    if not image_url and not image_base64:
        return {"error": "Must provide 'image_url' or 'image_base64'"}

    t0 = time.time()

    try:
        # Load image
        image_input = image_url or image_base64
        tile_image = load_image(image_input)
        h, w = tile_image.shape[:2]

        # Preprocess
        img_tensor = preprocess(tile_image)

        # Stage 1: Roof footprint
        roof_mask = predict_roof(img_tensor, (h, w))

        # Stage 2: Corrosion
        corrosion_mask = predict_corrosion(img_tensor, roof_mask, (h, w))

        # Compute areas
        roof_pixels = roof_mask.sum()
        corrosion_pixels = corrosion_mask.sum()
        roof_area_m2 = roof_pixels * gsd * gsd
        corroded_area_m2 = corrosion_pixels * gsd * gsd
        corrosion_percent = (corroded_area_m2 / roof_area_m2 * 100) if roof_area_m2 > 0 else 0.0

        severity = classify_severity(corrosion_percent)
        confidence = compute_confidence(img_tensor, corrosion_mask)

        elapsed_ms = (time.time() - t0) * 1000

        result = {
            "roof_area_m2": round(roof_area_m2, 2),
            "corroded_area_m2": round(corroded_area_m2, 2),
            "corrosion_percent": round(corrosion_percent, 1),
            "severity": severity,
            "confidence": round(confidence, 3),
            "model_versions": model_versions,
            "inference_time_ms": round(elapsed_ms, 1),
        }

        # Optionally include masks as base64 PNG
        if return_masks:
            roof_png = Image.fromarray((roof_mask * 255).astype(np.uint8))
            corr_png = Image.fromarray((corrosion_mask * 255).astype(np.uint8))
            buf_roof = io.BytesIO()
            buf_corr = io.BytesIO()
            roof_png.save(buf_roof, format="PNG")
            corr_png.save(buf_corr, format="PNG")
            result["roof_mask_b64"] = base64.b64encode(buf_roof.getvalue()).decode()
            result["corrosion_mask_b64"] = base64.b64encode(buf_corr.getvalue()).decode()

        print(f"Job {job_id}: severity={severity}, corrosion={corrosion_percent:.1f}%, confidence={confidence:.3f}, time={elapsed_ms:.0f}ms")
        return result

    except Exception as e:
        print(f"Job {job_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════

load_models()
runpod.serverless.start({"handler": handler})
