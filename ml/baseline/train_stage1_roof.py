"""Stage 1 baseline: Roof footprint segmentation with real models.

Architecture: SegFormer-B3 (primary) or Mask2Former-Swin-Small (alternative)
Pretrained on ADE20K, fine-tuned on AIRS/SpaceNet for binary roof segmentation.
Metric target: Roof IoU ≥ 0.85 on frozen real test set.

Usage:
    python ml/baseline/train_stage1_roof.py --config ml/baseline/configs/stage1_roof.yaml
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.baseline.augmentation import get_roof_augmentation
from ml.baseline.data.airs.loader import AIRSRoofDataset
from ml.baseline.data.spacenet.loader import SpaceNetBuildingDataset
from ml.baseline.models.roof_detector import RoofFootprintDetectorV2


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, dataloader, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training epoch {epoch}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        # SegFormer forward with built-in loss
        outputs = model(pixel_values=images, labels=masks)
        loss_ce = outputs.loss  # HuggingFace cross-entropy loss

        # Also compute Dice on the logits
        logits = outputs.logits  # (B, num_classes, H/4, W/4)
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = F.softmax(logits_up, dim=1)[:, 1]  # roof class probability
        loss_dice = dice_loss(probs, masks.float())

        loss = loss_ce + loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Compute IoU on validation set."""
    model.eval()
    intersection = 0
    union = 0
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        pred_labels = model.predict_mask(images)  # (B, H, W)

        intersection += (pred_labels * masks).sum().item()
        union += (pred_labels + masks).clamp(0, 1).sum().item()

    iou = intersection / max(union, 1)
    return iou


def main():
    parser = argparse.ArgumentParser(description="Stage 1 roof footprint baseline training")
    parser.add_argument("--config", type=str, default="ml/baseline/configs/stage1_roof.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment(f"stage1_roof_{cfg.dataset}")
    mlflow.start_run(tags={"data_sources": cfg.dataset, "phase": "1a-baseline"})

    # Log all config params
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    # Dataset selection with augmentation
    crop_size = cfg.get("augmentation", {}).get("crop_size", cfg.get("crop_size", 512))
    train_aug = get_roof_augmentation(crop_size, mode="train")
    val_aug = get_roof_augmentation(crop_size, mode="val")

    if cfg.dataset == "airs":
        train_ds = AIRSRoofDataset(
            image_dir=cfg.data.train_image_dir,
            mask_dir=cfg.data.train_mask_dir,
            transform=train_aug,
            crop_size=crop_size,
        )
        val_ds = AIRSRoofDataset(
            image_dir=cfg.data.val_image_dir,
            mask_dir=cfg.data.val_mask_dir,
            transform=val_aug,
            crop_size=crop_size,
        )
    elif cfg.dataset == "spacenet":
        train_ds = SpaceNetBuildingDataset(
            image_dir=cfg.data.train_image_dir,
            labels_geojson=cfg.data.train_labels,
            transform=train_aug,
            crop_size=crop_size,
        )
        val_ds = SpaceNetBuildingDataset(
            image_dir=cfg.data.val_image_dir,
            labels_geojson=cfg.data.val_labels,
            transform=val_aug,
            crop_size=crop_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # Model
    backbone = cfg.model.get("backbone", "b3")
    model = RoofFootprintDetectorV2(
        num_classes=cfg.model.num_classes,
        pretrained=True,
    ).to(device)

    # Two-phase training: freeze encoder first, then unfreeze
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineLR(optimizer, T_max=cfg.training.epochs)

    best_iou = 0.0
    warmup_epochs = cfg.training.get("warmup_epochs", 5)

    for epoch in range(cfg.training.epochs):
        # Unfreeze encoder after warmup
        if epoch == warmup_epochs:
            print(f"Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.lr * 0.1,  # lower LR for full fine-tune
                weight_decay=cfg.training.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineLR(optimizer, T_max=cfg.training.epochs - warmup_epochs)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_iou = validate(model, val_loader, device)
        scheduler.step()

        mlflow.log_metrics({"train_loss": train_loss, "val_iou": val_iou}, step=epoch)

        if val_iou > best_iou:
            best_iou = val_iou
            mlflow.pytorch.log_model(model.model, "best_model")

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_iou={val_iou:.4f}, best_iou={best_iou:.4f}")

    mlflow.log_metric("best_val_iou", best_iou)
    mlflow.end_run()

    print(f"\n{'='*60}")
    print(f"Stage 1 training complete. Best roof IoU: {best_iou:.4f}")
    print(f"Target: ≥ 0.85 on frozen real test set")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
