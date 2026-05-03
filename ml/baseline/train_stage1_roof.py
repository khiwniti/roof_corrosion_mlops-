"""Stage 1 baseline: Roof footprint segmentation.

Architecture: Mask2Former pretrained on SpaceNet → fine-tuned on AIRS
Metric target: Roof IoU ≥ 0.85 on frozen real test set

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

from ml.baseline.data.airs.loader import AIRSRoofDataset
from ml.baseline.data.spacenet.loader import SpaceNetBuildingDataset


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        # Forward pass — adapt depending on model head
        outputs = model(images)

        # Handle different output formats
        if hasattr(outputs, "masks"):
            pred = outputs.masks
        else:
            pred = outputs

        # Binary cross-entropy + Dice
        if pred.dim() == 4 and pred.shape[1] == 1:
            loss_bce = F.binary_cross_entropy_with_logits(pred.squeeze(1), masks.float())
            loss_dice = dice_loss(torch.sigmoid(pred.squeeze(1)), masks.float())
        elif pred.dim() == 4 and pred.shape[1] == 2:
            loss_bce = F.cross_entropy(pred, masks)
            loss_dice = dice_loss(
                F.softmax(pred, dim=1)[:, 1], masks.float()
            )
        else:
            loss_bce = F.cross_entropy(pred, masks)
            loss_dice = torch.tensor(0.0, device=device)

        loss = loss_bce + loss_dice
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

        outputs = model(images)
        if hasattr(outputs, "masks"):
            pred = outputs.masks
        else:
            pred = outputs

        if pred.dim() == 4 and pred.shape[1] == 2:
            pred_labels = pred.argmax(dim=1)
        elif pred.dim() == 4 and pred.shape[1] == 1:
            pred_labels = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
        else:
            pred_labels = pred.argmax(dim=1) if pred.dim() == 4 else pred

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

    # Dataset selection
    if cfg.dataset == "airs":
        train_ds = AIRSRoofDataset(
            image_dir=cfg.data.train_image_dir,
            mask_dir=cfg.data.train_mask_dir,
        )
        val_ds = AIRSRoofDataset(
            image_dir=cfg.data.val_image_dir,
            mask_dir=cfg.data.val_mask_dir,
        )
    elif cfg.dataset == "spacenet":
        train_ds = SpaceNetBuildingDataset(
            image_dir=cfg.data.train_image_dir,
            labels_geojson=cfg.data.train_labels,
        )
        val_ds = SpaceNetBuildingDataset(
            image_dir=cfg.data.val_image_dir,
            labels_geojson=cfg.data.val_labels,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    # Model — placeholder for Mask2Former; replace with actual init
    # from transformers import AutoModelForUniversalSegmentation
    # model = AutoModelForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-mh-22k")
    model = torch.nn.Identity()  # stub — replace with real model

    optimizer = torch.optim.AdamW(model.parameters() if hasattr(model, "parameters") else [], lr=cfg.lr)

    best_iou = 0.0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_iou = validate(model, val_loader, device)

        mlflow.log_metrics({"train_loss": train_loss, "val_iou": val_iou}, step=epoch)

        if val_iou > best_iou:
            best_iou = val_iou
            mlflow.pytorch.log_model(model, "best_model")

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_iou={val_iou:.4f}, best_iou={best_iou:.4f}")

    mlflow.log_metric("best_val_iou", best_iou)
    mlflow.end_run()


if __name__ == "__main__":
    main()
