"""Unified training script for roof corrosion segmentation pipeline.

Works with synthetic or real data. Runs on CPU or GPU.
Validates the full training → export → inference pipeline end-to-end.

Usage:
    # Train both stages on synthetic data (CPU-safe):
    python ml/train.py --data-dir data/synthetic --epochs 3 --batch-size 2

    # Train stage 1 only:
    python ml/train.py --stage 1 --data-dir data/synthetic --epochs 5

    # Train stage 2 only:
    python ml/train.py --stage 2 --data-dir data/synthetic --epochs 5

    # Export to TorchScript after training:
    python ml/train.py --data-dir data/synthetic --epochs 3 --export
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class RoofCorrosionDataset(Dataset):
    """Simple image/mask dataset from directory structure.

    Expected layout:
        data_dir/
            train/
                images/  synth_00000.png ...
                masks/   synth_00000.png ...
            val/
                images/  ...
                masks/   ...

    Mask classes: 0=background, 1=roof, 2=corrosion
    For Stage 1: roof=(mask>0).long() → binary {0,1}
    For Stage 2: mask as-is → 3-class {0,1,2}
    """

    def __init__(self, data_dir: str, split: str = "train", stage: int = 1, crop_size: int = 256):
        self.image_dir = Path(data_dir) / split / "images"
        self.mask_dir = Path(data_dir) / split / "masks"
        self.stage = stage
        self.crop_size = crop_size

        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            # Also try .jpg
            self.images = sorted(self.image_dir.glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.as_tensor(torch.from_numpy(__import__('numpy').array(mask)), dtype=torch.long)

        if self.stage == 1:
            # Binary: roof vs background
            mask = (mask > 0).long()

        return {"image": image, "mask": mask}


# ═══════════════════════════════════════════════════════════════
# Lightweight models (no HuggingFace dependency for CPU testing)
# ═══════════════════════════════════════════════════════════════

class LightweightSegModel(nn.Module):
    """Lightweight segmentation model for pipeline validation.

    This is NOT the production model — it's a small CNN that validates
    the training → export → inference pipeline works end-to-end.
    Production uses SegFormer-B3/B2 loaded from HuggingFace.

    Architecture: EfficientNet-style depthwise-separable encoder + FPN decoder.
    ~2M parameters, trains in minutes on CPU.
    """

    def __init__(self, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        self.num_classes = num_classes

        # Encoder (progressive downsampling: 1/2, 1/4, 1/8, 1/16)
        self.enc1 = self._ds_block(3, base_channels, stride=2)
        self.enc2 = self._ds_block(base_channels, base_channels * 2, stride=2)
        self.enc3 = self._ds_block(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = self._ds_block(base_channels * 4, base_channels * 8, stride=2)

        # Decoder (progressive upsampling with skip connections)
        self.dec4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels, 2, stride=2)
        self.dec1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )

    def _ds_block(self, in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        """Depthwise-separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Returns dict with 'logits' for compatibility with SegFormer API.
        """
        # Encoder
        e1 = self.enc1(x)    # 1/2
        e2 = self.enc2(e1)   # 1/4
        e3 = self.enc3(e2)   # 1/8
        e4 = self.enc4(e3)   # 1/16

        # Decoder with skip connections
        d4 = self.dec4(e4)                         # 1/8
        d4 = torch.cat([d4, e3], dim=1)            # skip from enc3
        d3 = self.dec3(d4)                          # 1/4
        d3 = torch.cat([d3, e2], dim=1)             # skip from enc2
        d2 = self.dec2(d3)                           # 1/2
        d2 = torch.cat([d2, e1], dim=1)              # skip from enc1
        d1 = self.dec1(d2)                            # 1/1

        logits = self.head(d1)                        # (B, C, H, W)

        return {"logits": logits}

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Predict segmentation mask."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output["logits"].argmax(dim=1)


# ═══════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def compute_loss(logits: torch.Tensor, masks: torch.Tensor, stage: int) -> torch.Tensor:
    """Compute combined CE + Dice (+ Focal for stage 2) loss."""
    # Cross-entropy
    loss_ce = F.cross_entropy(logits, masks)

    if stage == 1:
        # Binary dice on roof class
        probs = F.softmax(logits, dim=1)[:, 1]
        roof_mask = masks.float()
        loss_dice = dice_loss(probs, roof_mask)
        return loss_ce + loss_dice
    else:
        # Focal + Dice on corrosion class (class 2)
        probs = F.softmax(logits, dim=1)[:, 2]
        corrosion_mask = (masks == 2).float()
        loss_focal = focal_loss(logits[:, 2], corrosion_mask)
        loss_dice = dice_loss(probs, corrosion_mask)
        return loss_ce + loss_focal + loss_dice


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, device, stage):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        output = model(images)
        logits = output["logits"]

        loss = compute_loss(logits, masks, stage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model, dataloader, device, stage):
    """Compute IoU metrics."""
    model.eval()
    if stage == 1:
        # Binary roof IoU
        intersection, union = 0, 0
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            pred = model.predict_mask(images)
            intersection += (pred * masks).sum().item()
            union += (pred + masks).clamp(0, 1).sum().item()
        iou = intersection / max(union, 1)
        return {"iou": iou}
    else:
        # Corrosion IoU (class 2)
        tp, fp, fn = 0, 0, 0
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            corrosion_mask = (masks == 2).long()
            pred = model.predict_mask(images)
            pred_corrosion = (pred == 2).long()
            tp += (pred_corrosion * corrosion_mask).sum().item()
            fp += (pred_corrosion * (1 - corrosion_mask)).sum().item()
            fn += ((1 - pred_corrosion) * corrosion_mask).sum().item()
        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {"iou": iou, "precision": precision, "recall": recall}


def export_torchscript(model, output_path: str, crop_size: int = 256):
    """Export model to TorchScript for production serving."""
    model.eval()
    dummy = torch.randn(1, 3, crop_size, crop_size)
    scripted = torch.jit.trace(model, dummy)

    # Verify the scripted model produces same output
    with torch.no_grad():
        orig_out = model(dummy)["logits"]
        script_out = scripted(dummy)
        if isinstance(script_out, dict):
            script_out = script_out["logits"]
        max_diff = (orig_out - script_out).abs().max().item()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))

    size_mb = output.stat().st_size / 1e6
    print(f"  TorchScript model saved to {output} ({size_mb:.1f} MB)")
    print(f"  Max diff vs original: {max_diff:.6f}")
    return str(output)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train roof corrosion segmentation models")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Train only stage 1 or 2. Default: train both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--export", action="store_true", help="Export to TorchScript after training")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")

    stages = [args.stage] if args.stage else [1, 2]
    results = {}

    for stage in stages:
        num_classes = 2 if stage == 1 else 3
        stage_name = "roof_footprint" if stage == 1 else "corrosion"
        print(f"\n{'='*60}")
        print(f"  Stage {stage}: {stage_name} segmentation ({num_classes} classes)")
        print(f"{'='*60}")

        # Dataset
        train_ds = RoofCorrosionDataset(args.data_dir, "train", stage=stage, crop_size=args.crop_size)
        val_ds = RoofCorrosionDataset(args.data_dir, "val", stage=stage, crop_size=args.crop_size)
        print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Model
        model = LightweightSegModel(num_classes=num_classes).to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model: LightweightSegModel ({param_count:.2f}M params)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_iou = 0.0
        best_state = None

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, stage)
            metrics = validate(model, val_loader, device, stage)
            scheduler.step()
            elapsed = time.time() - t0

            iou = metrics["iou"]
            if iou > best_iou:
                best_iou = iou
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            extra = ""
            if stage == 2:
                extra = f", prec={metrics['precision']:.4f}, rec={metrics['recall']:.4f}"
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, iou={iou:.4f}{extra}, best_iou={best_iou:.4f} ({elapsed:.1f}s)")

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Export
        model_path = None
        if args.export:
            model_path = export_torchscript(
                model,
                str(Path(args.output_dir) / f"stage{stage}_{stage_name}.pt"),
                crop_size=args.crop_size,
            )

        results[f"stage{stage}"] = {
            "best_iou": best_iou,
            "num_classes": num_classes,
            "model_path": model_path,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    for stage_key, r in results.items():
        print(f"  {stage_key}: best_iou={r['best_iou']:.4f}, model={r['model_path']}")

    # Save results
    results_path = Path(args.output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Go/no-go gate for stage 2
    if "stage2" in results:
        s2_iou = results["stage2"]["best_iou"]
        print(f"\n  GO/NO-GO GATE (Phase 1a):")
        if s2_iou >= 0.45:
            print(f"  GO: IoU {s2_iou:.4f} >= 0.45 — proceed to Phase 1b")
        elif s2_iou >= 0.25:
            print(f"  CONDITIONAL: IoU {s2_iou:.4f} in [0.25, 0.45) — full synthetic + DA plan")
        else:
            print(f"  NO-GO: IoU {s2_iou:.4f} < 0.25 — revisit task framing")
        print(f"  (Note: synthetic data IoU is not representative of real performance)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
