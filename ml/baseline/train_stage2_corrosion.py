"""Stage 2 baseline: Corrosion segmentation on roof crops.

Architecture: SegFormer-B2 fine-tuned on Caribbean irregular_metal class
Metric target: Corrosion IoU ≥ 0.45 on frozen real test set (go/no-go gate)

Usage:
    python ml/baseline/train_stage2_corrosion.py --config ml/baseline/configs/stage2_corrosion.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.baseline.data.caribbean.loader import CaribbeanRoofDataset


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for handling class imbalance (corrosion is rare)."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        # Corrosion mask: class 2 = corroded, everything else = not corroded
        corrosion_mask = (masks == 2).float()

        outputs = model(images)
        if hasattr(outputs, "logits"):
            pred = outputs.logits
        else:
            pred = outputs

        # Resize pred to match mask if needed
        if pred.shape[-2:] != corrosion_mask.shape[-2:]:
            pred = F.interpolate(pred, size=corrosion_mask.shape[-2:], mode="bilinear", align_corners=False)

        if pred.dim() == 4 and pred.shape[1] == 1:
            loss_focal = focal_loss(pred.squeeze(1), corrosion_mask)
            loss_dice = dice_loss(torch.sigmoid(pred.squeeze(1)), corrosion_mask)
        elif pred.dim() == 4 and pred.shape[1] == 2:
            # Two-class: [background, corrosion]
            loss_focal = F.cross_entropy(pred, corrosion_mask.long())
            loss_dice = dice_loss(F.softmax(pred, dim=1)[:, 1], corrosion_mask)
        else:
            loss_focal = F.cross_entropy(pred, corrosion_mask.long())
            loss_dice = torch.tensor(0.0, device=device)

        loss = loss_focal + loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Compute corrosion IoU and per-pixel precision/recall."""
    model.eval()
    tp, fp, fn = 0, 0, 0
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()
        corrosion_mask = (masks == 2).long()

        outputs = model(images)
        if hasattr(outputs, "logits"):
            pred = outputs.logits
        else:
            pred = outputs

        if pred.shape[-2:] != corrosion_mask.shape[-2:]:
            pred = F.interpolate(pred, size=corrosion_mask.shape[-2:], mode="bilinear", align_corners=False)

        if pred.dim() == 4 and pred.shape[1] == 2:
            pred_labels = pred.argmax(dim=1)
        elif pred.dim() == 4 and pred.shape[1] == 1:
            pred_labels = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
        else:
            pred_labels = pred.argmax(dim=1) if pred.dim() == 4 else pred

        tp += (pred_labels * corrosion_mask).sum().item()
        fp += (pred_labels * (1 - corrosion_mask)).sum().item()
        fn += ((1 - pred_labels) * corrosion_mask).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return {"iou": iou, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser(description="Stage 2 corrosion segmentation baseline training")
    parser.add_argument("--config", type=str, default="ml/baseline/configs/stage2_corrosion.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment("stage2_corrosion_caribbean")
    mlflow.start_run(tags={"data_sources": "caribbean", "phase": "1a-baseline"})

    train_ds = CaribbeanRoofDataset(
        image_dir=cfg.data.train_image_dir,
        labels_geojson=cfg.data.train_labels,
    )
    val_ds = CaribbeanRoofDataset(
        image_dir=cfg.data.val_image_dir,
        labels_geojson=cfg.data.val_labels,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Model — placeholder for SegFormer-B2; replace with actual init
    # from transformers import SegformerForSemanticSegmentation
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/segformer-b2-finetuned-ade-512-512",
    #     num_labels=2,
    #     ignore_mismatched_sizes=True,
    # )
    model = torch.nn.Identity()  # stub — replace with real model

    optimizer = torch.optim.AdamW(model.parameters() if hasattr(model, "parameters") else [], lr=cfg.lr)

    best_iou = 0.0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = validate(model, val_loader, device)

        mlflow.log_metrics(
            {"train_loss": train_loss, "val_iou": metrics["iou"],
             "val_precision": metrics["precision"], "val_recall": metrics["recall"]},
            step=epoch,
        )

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            mlflow.pytorch.log_model(model, "best_model")

        print(
            f"Epoch {epoch}: loss={train_loss:.4f}, "
            f"iou={metrics['iou']:.4f}, prec={metrics['precision']:.4f}, "
            f"rec={metrics['recall']:.4f}, best_iou={best_iou:.4f}"
        )

    mlflow.log_metric("best_val_iou", best_iou)
    mlflow.end_run()

    # ── Go/no-go gate ────────────────────────────────────────
    GO_NOGO_IOU_THRESHOLD = 0.45
    if best_iou >= GO_NOGO_IOU_THRESHOLD:
        print(f"\n✅ GO: Baseline corrosion IoU {best_iou:.4f} ≥ {GO_NOGO_IOU_THRESHOLD}. Proceed to Phase 1b.")
    elif best_iou >= 0.25:
        print(f"\n⚠️  CONDITIONAL: IoU {best_iou:.4f} in [0.25, 0.45). Proceed with synthetic + DA plan.")
    else:
        print(f"\n❌ NO-GO: IoU {best_iou:.4f} < 0.25. Revisit task framing before further investment.")


if __name__ == "__main__":
    main()
