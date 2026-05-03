"""Stage 2 baseline: Corrosion segmentation with real SegFormer-B2.

Fine-tuned on Caribbean irregular_metal class as corrosion proxy.
Uses Focal + Dice loss to handle severe class imbalance.
Metric target: Corrosion IoU ≥ 0.45 on open-data baseline (go/no-go gate).

Usage:
    python ml/baseline/train_stage2_corrosion.py --config ml/baseline/configs/stage2_corrosion.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.baseline.augmentation import get_corrosion_augmentation
from ml.baseline.data.caribbean.loader import CaribbeanRoofDataset
from ml.baseline.models.corrosion_detector import CorrosionDetector


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


def train_one_epoch(model, dataloader, optimizer, device, epoch: int, cfg):
    model.train()
    total_loss = 0.0
    focal_alpha = cfg.training.get("focal_alpha", 0.25)
    focal_gamma = cfg.training.get("focal_gamma", 2.0)

    for batch in tqdm(dataloader, desc=f"Training epoch {epoch}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        # SegFormer forward with built-in CE loss
        outputs = model(pixel_values=images, labels=masks)
        loss_ce = outputs.loss

        # Additional Focal + Dice on corrosion class
        logits = outputs.logits  # (B, 3, H/4, W/4)
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = F.softmax(logits_up, dim=1)[:, 2]  # corrosion probability
        corrosion_mask = (masks == 2).float()

        loss_focal = focal_loss(logits_up[:, 2], corrosion_mask, alpha=focal_alpha, gamma=focal_gamma)
        loss_dice = dice_loss(probs, corrosion_mask)

        loss = loss_ce + loss_focal + loss_dice
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

        pred_labels = model.predict_mask(images)  # (B, H, W)

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
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    # Dataset with augmentation
    crop_size = cfg.get("augmentation", {}).get("crop_size", cfg.get("crop_size", 512))
    train_aug = get_corrosion_augmentation(crop_size, mode="train")
    val_aug = get_corrosion_augmentation(crop_size, mode="val")

    train_ds = CaribbeanRoofDataset(
        image_dir=cfg.data.train_image_dir,
        labels_geojson=cfg.data.train_labels,
        transform=train_aug,
        crop_size=crop_size,
    )
    val_ds = CaribbeanRoofDataset(
        image_dir=cfg.data.val_image_dir,
        labels_geojson=cfg.data.val_labels,
        transform=val_aug,
        crop_size=crop_size,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # Model: SegFormer-B2
    backbone = cfg.model.get("backbone", "b2")
    model = CorrosionDetector(
        backbone=backbone,
        num_classes=cfg.model.num_classes,
        pretrained=True,
        freeze_encoder=True,  # freeze encoder for first N epochs
    ).to(device)

    # Two-phase training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineLR(optimizer, T_max=cfg.training.epochs)

    best_iou = 0.0
    warmup_epochs = cfg.training.get("warmup_epochs", 5)

    for epoch in range(cfg.training.epochs):
        if epoch == warmup_epochs:
            print(f"Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.lr * 0.1,
                weight_decay=cfg.training.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineLR(
                optimizer, T_max=cfg.training.epochs - warmup_epochs
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
        metrics = validate(model, val_loader, device)
        scheduler.step()

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_iou": metrics["iou"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            },
            step=epoch,
        )

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            mlflow.pytorch.log_model(model.model, "best_model")

        print(
            f"Epoch {epoch}: loss={train_loss:.4f}, "
            f"iou={metrics['iou']:.4f}, prec={metrics['precision']:.4f}, "
            f"rec={metrics['recall']:.4f}, best_iou={best_iou:.4f}"
        )

    mlflow.log_metric("best_val_iou", best_iou)
    mlflow.end_run()

    # ── Go/no-go gate ────────────────────────────────────────
    GO_NOGO_IOU_THRESHOLD = 0.45
    CONDITIONAL_THRESHOLD = 0.25

    print(f"\n{'='*60}")
    print(f"GO / NO-GO GATE — Phase 1a Baseline")
    print(f"{'='*60}")

    if best_iou >= GO_NOGO_IOU_THRESHOLD:
        print(f"✅ GO: Baseline corrosion IoU {best_iou:.4f} ≥ {GO_NOGO_IOU_THRESHOLD}")
        print(f"   Proceed to Phase 1b with light labeling only.")
    elif best_iou >= CONDITIONAL_THRESHOLD:
        print(f"⚠️  CONDITIONAL: IoU {best_iou:.4f} in [{CONDITIONAL_THRESHOLD}, {GO_NOGO_IOU_THRESHOLD})")
        print(f"   Proceed with full synthetic + domain adaptation plan.")
    else:
        print(f"❌ NO-GO: IoU {best_iou:.4f} < {CONDITIONAL_THRESHOLD}")
        print(f"   Revisit task framing before further investment.")
        print(f"   Consider: severity grading instead of segmentation,")
        print(f"   or pivot to drone-only product.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
