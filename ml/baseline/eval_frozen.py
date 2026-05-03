"""Frozen real-image test set evaluation harness.

This script evaluates ANY model against the immutable frozen test set.
The frozen set is NEVER used in training and NEVER regenerated.
It is versioned in DVC and checksummed in CI.

Usage:
    python ml/baseline/eval_frozen.py \
        --model-uri mlflow:///models/stage2_corrosion/1 \
        --frozen-dir data/frozen_test/ \
        --output-dir ml/baseline/frozen_eval_results/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class FrozenTestDataset(torch.utils.data.Dataset):
    """Generic frozen test set loader. Expects .tif tiles + .tif masks."""

    def __init__(self, frozen_dir: str | Path, transform=None):
        self.frozen_dir = Path(frozen_dir)
        self.transform = transform
        self.image_dir = self.frozen_dir / "images"
        self.mask_dir = self.frozen_dir / "masks"
        self.tiles = sorted(self.image_dir.glob("*.tif"))
        if not self.tiles:
            raise FileNotFoundError(f"No .tif tiles in {self.image_dir}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        import rasterio

        tile_path = self.tiles[idx]
        tile_id = tile_path.stem

        with rasterio.open(tile_path) as src:
            img = src.read([1, 2, 3]).astype(np.float32) / 255.0
            gsd = src.res[0]

        mask_path = self.mask_dir / f"{tile_id}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        if self.transform:
            transformed = self.transform(image=np.transpose(img, (1, 2, 0)), mask=mask)
            img = np.transpose(transformed["image"], (2, 0, 1))
            mask = transformed["mask"]

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "metadata": {"tile_id": tile_id, "gsd": gsd},
        }


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes: int = 3):
    """Compute comprehensive metrics on the frozen test set.

    Classes: 0=background, 1=roof, 2=corrosion
    """
    model.eval()
    class_tp = np.zeros(num_classes)
    class_fp = np.zeros(num_classes)
    class_fn = np.zeros(num_classes)
    area_errors = []

    for batch in tqdm(dataloader, desc="Evaluating on frozen set"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()
        gsds = [m["gsd"] for m in batch["metadata"]]

        outputs = model(images)
        if hasattr(outputs, "logits"):
            pred = outputs.logits
        elif hasattr(outputs, "masks"):
            pred = outputs.masks
        else:
            pred = outputs

        if pred.shape[-2:] != masks.shape[-2:]:
            pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        if pred.dim() == 4 and pred.shape[1] > 1:
            pred_labels = pred.argmax(dim=1)
        elif pred.dim() == 4 and pred.shape[1] == 1:
            pred_labels = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
        else:
            pred_labels = pred

        # Per-class IoU
        for c in range(num_classes):
            pred_c = (pred_labels == c)
            true_c = (masks == c)
            class_tp[c] += (pred_c & true_c).sum().item()
            class_fp[c] += (pred_c & ~true_c).sum().item()
            class_fn[c] += (~pred_c & true_c).sum().item()

        # Area MAPE for corrosion (class 2)
        for i in range(len(gsds)):
            gsd = gsds[i]
            pred_corr = (pred_labels[i] == 2).sum().item()
            true_corr = (masks[i] == 2).sum().item()
            pred_area = pred_corr * gsd * gsd
            true_area = true_corr * gsd * gsd
            if true_area > 0:
                area_errors.append(abs(pred_area - true_area) / true_area)

    results = {}
    class_names = ["background", "roof", "corrosion"]
    for c, name in enumerate(class_names):
        iou = class_tp[c] / max(class_tp[c] + class_fp[c] + class_fn[c], 1)
        precision = class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        recall = class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        results[f"{name}_iou"] = float(iou)
        results[f"{name}_precision"] = float(precision)
        results[f"{name}_recall"] = float(recall)

    results["area_mape"] = float(np.mean(area_errors)) if area_errors else float("nan")
    results["area_mape_median"] = float(np.median(area_errors)) if area_errors else float("nan")
    results["num_tiles"] = len(dataloader.dataset)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on frozen real-image test set")
    parser.add_argument("--model-uri", type=str, required=True, help="MLflow model URI")
    parser.add_argument("--frozen-dir", type=str, default="data/frozen_test")
    parser.add_argument("--output-dir", type=str, default="ml/baseline/frozen_eval_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model from MLflow
    model = mlflow.pytorch.load_model(args.model_uri)
    model = model.to(device)

    # Load frozen test set
    dataset = FrozenTestDataset(args.frozen_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Evaluate
    results = evaluate_model(model, dataloader, device)
    results["model_uri"] = args.model_uri
    results["evaluated_at"] = datetime.utcnow().isoformat()

    # Save results
    results_path = output_dir / "frozen_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FROZEN TEST SET EVALUATION RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # ── Promotion gate checks ────────────────────────────────
    print("\n" + "=" * 60)
    print("PROMOTION GATE CHECKS")
    print("=" * 60)

    gates = {
        "roof_iou ≥ 0.85": results.get("roof_iou", 0) >= 0.85,
        "corrosion_iou ≥ 0.55": results.get("corrosion_iou", 0) >= 0.55,
        "area_mape ≤ 15%": results.get("area_mape", float("inf")) <= 0.15,
    }

    all_pass = True
    for gate_name, passed in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {gate_name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🟢 Model eligible for production promotion.")
    else:
        print("\n🔴 Model does NOT meet production promotion criteria.")


if __name__ == "__main__":
    main()
