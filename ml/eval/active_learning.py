"""Active learning sampler for roof corrosion model improvement.

Selects the most informative unlabeled tiles for human annotation,
using uncertainty-based sampling strategies:
1. Monte Carlo dropout disagreement (entropy)
2. Model prediction margin (low-confidence predictions)
3. Diversity sampling (cover feature space evenly)

Outputs a list of tile IDs to send to Label Studio for HITL labeling.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ml.baseline.models.corrosion_detector import CorrosionDetector, mc_dropout_uncertainty


# ── Sampling strategies ─────────────────────────────────────

def uncertainty_sample(
    model: CorrosionDetector,
    tile_images: list[np.ndarray],
    tile_ids: list[str],
    budget: int = 50,
    num_mc_samples: int = 10,
    device: str = "cuda",
) -> list[dict]:
    """Select tiles with highest model uncertainty (MC-dropout entropy).

    Args:
        model: trained CorrosionDetector
        tile_images: list of (H, W, 3) uint8 arrays
        tile_ids: corresponding tile identifiers
        budget: number of tiles to select
        num_mc_samples: MC dropout forward passes
        device: torch device

    Returns:
        List of {tile_id, entropy_mean, entropy_max, rank} sorted by uncertainty
    """
    model = model.to(device)
    model.eval()

    results = []
    for i, (img, tid) in enumerate(zip(tile_images, tile_ids)):
        # Preprocess
        img_tensor = preprocess_image(img, device)

        # MC-dropout uncertainty
        uncertainty = mc_dropout_uncertainty(model, img_tensor, num_samples=num_mc_samples)
        entropy = uncertainty["entropy"].cpu().numpy()  # (H, W)

        results.append({
            "tile_id": tid,
            "entropy_mean": float(entropy.mean()),
            "entropy_max": float(entropy.max()),
            "entropy_p95": float(np.percentile(entropy, 95)),
            "index": i,
        })

    # Sort by mean entropy (highest uncertainty first)
    results.sort(key=lambda x: x["entropy_mean"], reverse=True)

    # Select top-k
    selected = results[:budget]
    for rank, item in enumerate(selected):
        item["rank"] = rank + 1
        item["strategy"] = "uncertainty_entropy"

    return selected


def margin_sample(
    model: CorrosionDetector,
    tile_images: list[np.ndarray],
    tile_ids: list[str],
    budget: int = 50,
    device: str = "cuda",
) -> list[dict]:
    """Select tiles where model is least confident (smallest margin between top-2 classes).

    Useful when the model is uncertain between roof and corrosion classes.
    """
    model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for i, (img, tid) in enumerate(zip(tile_images, tile_ids)):
            img_tensor = preprocess_image(img, device)
            outputs = model(pixel_values=img_tensor)
            logits = outputs.logits
            logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (1, 3, H, W)

            # Margin: difference between top-2 class probabilities
            sorted_probs = np.sort(probs[0], axis=0)  # (3, H, W) ascending
            margin = sorted_probs[-1] - sorted_probs[-2]  # (H, W)
            min_margin = float(margin.min())
            mean_margin = float(margin.mean())

            # Confidence of corrosion class
            corrosion_prob = float(probs[0, 2].mean())

            results.append({
                "tile_id": tid,
                "min_margin": min_margin,
                "mean_margin": mean_margin,
                "corrosion_prob_mean": corrosion_prob,
                "index": i,
            })

    # Sort by smallest margin (most uncertain)
    results.sort(key=lambda x: x["min_margin"])

    selected = results[:budget]
    for rank, item in enumerate(selected):
        item["rank"] = rank + 1
        item["strategy"] = "margin_sampling"

    return selected


def diversity_sample(
    features: np.ndarray,
    tile_ids: list[str],
    budget: int = 50,
    existing_features: Optional[np.ndarray] = None,
) -> list[dict]:
    """Select tiles that are maximally different from already-labeled data.

    Uses k-means++ initialization to cover feature space evenly.
    Features should be extracted from a trained encoder backbone.

    Args:
        features: (N, D) feature vectors for each tile
        tile_ids: corresponding tile IDs
        budget: number to select
        existing_features: (M, D) features of already-labeled tiles
    """
    from sklearn.cluster import KMeans

    n = len(tile_ids)
    if n <= budget:
        return [{"tile_id": tid, "rank": i + 1, "strategy": "diversity"} for i, tid in enumerate(tile_ids)]

    # If we have existing labeled data, find tiles farthest from them
    if existing_features is not None and len(existing_features) > 0:
        # Compute distance to nearest labeled sample
        dists = np.zeros(n)
        for i in range(n):
            min_dist = np.min(np.linalg.norm(existing_features - features[i], axis=1))
            dists[i] = min_dist

        # Select top-k farthest
        indices = np.argsort(dists)[::-1][:budget]
    else:
        # K-means++ to find diverse centroids, then pick nearest tile to each
        k = min(budget, n)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(features)
        _, indices = kmeans.predict(features), None

        # For each centroid, find nearest tile
        distances = kmeans.transform(features)  # (N, k)
        indices = []
        for c in range(k):
            nearest = np.argmin(distances[:, c])
            if nearest not in indices:
                indices.append(nearest)

        indices = indices[:budget]

    return [
        {"tile_id": tile_ids[idx], "rank": rank + 1, "strategy": "diversity"}
        for rank, idx in enumerate(indices)
    ]


def combined_sample(
    model: CorrosionDetector,
    tile_images: list[np.ndarray],
    tile_ids: list[str],
    budget: int = 50,
    uncertainty_weight: float = 0.5,
    diversity_weight: float = 0.3,
    margin_weight: float = 0.2,
    device: str = "cuda",
) -> list[dict]:
    """Combined sampling: weighted union of uncertainty, margin, and diversity.

    Default weights prioritize uncertainty (most effective for segmentation),
    with diversity to avoid redundant labeling of similar tiles.
    """
    # Get scores from each strategy
    unc_samples = uncertainty_sample(model, tile_images, tile_ids, budget=budget * 2, device=device)
    margin_samples = margin_sample(model, tile_images, tile_ids, budget=budget * 2, device=device)

    # Build score maps (tile_id → normalized score)
    unc_scores = {s["tile_id"]: 1.0 / s["rank"] for s in unc_samples}
    margin_scores = {s["tile_id"]: 1.0 / s["rank"] for s in margin_samples}

    # Combined score
    all_ids = set(unc_scores.keys()) | set(margin_scores.keys())
    combined = []
    for tid in all_ids:
        score = (
            uncertainty_weight * unc_scores.get(tid, 0)
            + margin_weight * margin_scores.get(tid, 0)
        )
        combined.append({"tile_id": tid, "combined_score": score, "strategy": "combined"})

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    selected = combined[:budget]

    for rank, item in enumerate(selected):
        item["rank"] = rank + 1

    return selected


# ── Label Studio integration ────────────────────────────────

def export_to_label_studio(
    selected_tiles: list[dict],
    tile_images: list[np.ndarray],
    tile_ids: list[str],
    project_id: int = 1,
) -> dict:
    """Export selected tiles to Label Studio for HITL labeling.

    Creates a labeling task for each selected tile with pre-filled
    model predictions as initial annotations (for correction).
    """
    import os

    label_studio_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    label_studio_key = os.getenv("LABEL_STUDIO_API_KEY", "")

    # Build Label Studio import JSON
    tasks = []
    id_to_idx = {tid: i for i, tid in enumerate(tile_ids)}

    for item in selected_tiles:
        tid = item["tile_id"]
        idx = id_to_idx.get(tid)
        if idx is None:
            continue

        tasks.append({
            "data": {
                "tile_id": tid,
                "image": f"/data/local-files/?d=tiles/{tid}.tif",
                "sampling_strategy": item.get("strategy", "unknown"),
                "sampling_rank": item.get("rank", 0),
                "uncertainty_score": item.get("combined_score", item.get("entropy_mean", 0)),
            },
            "predictions": [],  # TODO: add model predictions as initial annotations
        })

    # Write export file
    export_path = Path("data/active_learning/label_studio_import.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Exported {len(tasks)} tasks to {export_path}")
    print(f"Import into Label Studio: {label_studio_url}/projects/{project_id}/")

    return {"tasks_created": len(tasks), "export_path": str(export_path)}


# ── Utilities ───────────────────────────────────────────────

def preprocess_image(img: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Convert (H, W, 3) uint8 image to (1, 3, H, W) normalized tensor."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert to PIL for torchvision transforms
    from PIL import Image
    pil_img = Image.fromarray(img)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor


if __name__ == "__main__":
    print("Active learning sampler module.")
    print("Usage: Import and call uncertainty_sample() or combined_sample()")
    print("       then export_to_label_studio() to create HITL tasks.")
