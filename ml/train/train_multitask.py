"""Main training script for Clay + Mask2Former multi-task model.

Usage:
    python ml/train/train_multitask.py --config ml/train/configs/multitask_clay.yaml

Architecture Decision: ADR-002
Training Recipe: SPEC-phase-2 §6
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.train.models.clay_multitask import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_multitask")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict) -> dict:
    """Build train/val dataloaders from config."""
    # Stub: return empty dict — implement per-stage dataset selection
    logger.info("Building dataloaders (stub) — datasets: %s", config.get("data", {}).keys())
    return {}


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Layer-wise LR optimizer per ADR-002 recipe."""
    opt_cfg = config["training"]["optimizer"]
    params = [
        {"params": [], "lr": opt_cfg["lr_backbone"], "name": "backbone"},
        {"params": [], "lr": opt_cfg["lr_adapter"], "name": "adapter"},
        {"params": [], "lr": opt_cfg["lr_decoder"], "name": "decoder"},
        {"params": [], "lr": opt_cfg["lr_decoder"], "name": "heads"},
    ]
    # Assign parameters to groups (simplified — real impl inspects module names)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "material_head" in name or "corrosion_head" in name or "severity_head" in name:
            params[3]["params"].append(p)
        elif "pixel_decoder" in name or "query" in name:
            params[2]["params"].append(p)
        elif "stub_encoder" in name:
            params[0]["params"].append(p)
        else:
            params[1]["params"].append(p)

    # Remove empty groups
    params = [g for g in params if len(g["params"]) > 0]

    return torch.optim.AdamW(params, weight_decay=opt_cfg["weight_decay"])


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    sched_cfg = config["training"]["scheduler"]
    total_epochs = sum(s["epochs"] for s in config["training"]["stages"])
    warmup = config["training"]["stages"][0].get("warmup_epochs", sched_cfg.get("warmup_epochs", 5))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup)


def material_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
) -> torch.Tensor:
    """Dice + Cross-Entropy loss for material segmentation."""
    ce = nn.functional.cross_entropy(logits, target, ignore_index=255)
    # Simplified Dice
    pred = torch.softmax(logits, dim=1)
    target_onehot = nn.functional.one_hot(target.clamp(0), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = 1 - (2 * intersection / (union + 1e-6)).mean()
    return dice_weight * dice + ce_weight * ce


def corrosion_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for corrosion binary segmentation."""
    ce = nn.functional.cross_entropy(logits, target, reduction="none", ignore_index=255)
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma * ce).mean()
    return focal


def severity_ordinal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Ordinal regression loss for severity levels."""
    # Cumulative logits approach: P(y > k)
    probs = torch.softmax(logits, dim=1)
    target_onehot = nn.functional.one_hot(target.clamp(0), num_classes=logits.shape[1]).float()
    # Cumulative targets
    cum_target = torch.cumsum(target_onehot, dim=1)
    cum_pred = torch.cumsum(probs, dim=1)
    # Binary CE on cumulative probabilities
    bce = nn.functional.binary_cross_entropy(cum_pred, cum_target, reduction="mean")
    return bce


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Stub: assume batch has keys needed for each head
        # Real impl unpacks per-stage dataset format
        optimizer.zero_grad()
        # Forward (stub — batch structure varies by stage)
        # loss = compute_loss(model, batch, config)
        # loss.backward()
        # optimizer.step()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    # Stub: compute mIoU for each head
    return {"val_material_miou": 0.0, "val_corrosion_miou": 0.0, "val_severity_acc": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Clay multi-task model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--stage", default="roofnet_pretrain", help="Training stage name")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    logger.info("Starting stage: %s", args.stage)
    logger.info("Config: %s", args.config)

    device = torch.device(config.get("hardware", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # Build model
    model = build_model(config)
    model.to(device)

    # Load pretrained Clay weights if available
    clay_ckpt = config.get("model", {}).get("clay_checkpoint")
    if clay_ckpt and os.path.exists(clay_ckpt):
        model.load_clay_weights(clay_ckpt)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Build dataloaders
    dataloaders = build_dataloaders(config)
    logger.info("Dataloaders: %s", list(dataloaders.keys()))

    # Training loop (stub)
    logger.info("Training loop ready — implement per-stage dataset selection and loss computation.")
    logger.info(
        "Expected stages: %s",
        [s["name"] for s in config["training"]["stages"]],
    )

    # Save a dummy checkpoint to verify save path
    out_dir = Path("outputs") / config.get("experiment_name", "clay_multitask")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "stage": args.stage,
    }, out_dir / "checkpoint_last.pth")
    logger.info("Saved stub checkpoint to %s", out_dir)


if __name__ == "__main__":
    main()
