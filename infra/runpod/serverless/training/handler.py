"""RunPod Serverless training handler for roof corrosion models.

Runs Stage 1 (roof) or Stage 2 (corrosion) training on GPU, then
uploads TorchScript artifacts to S3 for the inference endpoint.

Deploy:
    docker build -t roof-corrosion-training .
    docker push YOUR_REGISTRY/roof-corrosion-training:latest

Local test:
    python handler.py --test_input test_input.json

API request format:
    POST /run
    {
        "input": {
            "stage": 1,                     # 1=roof, 2=corrosion
            "epochs": 50,
            "batch_size": 4,
            "lr": 1e-4,
            "backbone": "b3",               # SegFormer backbone size
            "data_source": "s3",             # s3 | synthetic | local
            "s3_data_path": "s3://bucket/data/",
            "s3_output_path": "s3://bucket/models/",
            "upload_artifacts": true          # upload TorchScript to S3
        }
    }
"""

import json
import os
import sys
import time
from pathlib import Path

import runpod
import torch


# ═══════════════════════════════════════════════════════════════
# Training logic
# ═══════════════════════════════════════════════════════════════

def run_training(job_input: dict) -> dict:
    """Execute model training and return metrics + artifact paths."""
    stage = job_input.get("stage", 1)
    epochs = job_input.get("epochs", 50)
    batch_size = job_input.get("batch_size", 4)
    lr = job_input.get("lr", 1e-4)
    backbone = job_input.get("backbone", "b3" if stage == 1 else "b2")
    data_source = job_input.get("data_source", "synthetic")
    upload_artifacts = job_input.get("upload_artifacts", True)
    s3_output_path = job_input.get("s3_output_path", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Stage {stage} on {device}, backbone={backbone}, epochs={epochs}")

    # ── Prepare data ─────────────────────────────────────────
    data_dir = Path("/app/data")
    if data_source == "s3":
        s3_data_path = job_input.get("s3_data_path", "")
        if s3_data_path:
            print(f"Downloading data from {s3_data_path}...")
            os.system(f"aws s3 sync {s3_data_path} {data_dir} --no-sign-request 2>/dev/null || "
                      f"aws s3 sync {s3_data_path} {data_dir}")
    elif data_source == "synthetic":
        # Generate synthetic training data on the fly
        print("Generating synthetic dataset...")
        sys.path.insert(0, "/app/ml")
        from synth.generate_synthetic import generate_dataset
        generate_dataset(num_tiles=200, output_dir=str(data_dir / "synthetic"), tile_size=512)
        data_dir = data_dir / "synthetic"

    # ── Import model and training code ───────────────────────
    sys.path.insert(0, "/app/ml")
    from baseline.models.roof_detector import RoofFootprintDetectorV2
    from baseline.models.corrosion_detector import CorrosionDetector

    num_classes = 2 if stage == 1 else 3
    stage_name = "roof_footprint" if stage == 1 else "corrosion"

    # ── Build model ──────────────────────────────────────────
    if stage == 1:
        model = RoofFootprintDetectorV2(num_classes=num_classes, pretrained=True)
    else:
        model = CorrosionDetector(backbone=backbone, num_classes=num_classes,
                                   pretrained=True, freeze_encoder=True)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters")

    # ── Build dataset ────────────────────────────────────────
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, mask_dir, stage, crop_size=512):
            self.images = sorted(Path(img_dir).glob("*.png"))
            if not self.images:
                self.images = sorted(Path(img_dir).glob("*.jpg"))
            self.mask_dir = Path(mask_dir)
            self.stage = stage
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_resize = transforms.Resize((crop_size, crop_size),
                                                  interpolation=transforms.InterpolationMode.NEAREST)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert("RGB")
            mask = Image.open(self.mask_dir / self.images[idx].name)
            img = self.transform(img)
            mask = self.mask_resize(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            if self.stage == 1:
                mask = (mask > 0).long()
            return {"image": img, "mask": mask}

    train_ds = SimpleDataset(data_dir / "train" / "images", data_dir / "train" / "masks", stage)
    val_ds = SimpleDataset(data_dir / "val" / "images", data_dir / "val" / "masks", stage)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Training loop ────────────────────────────────────────
    import torch.nn.functional as F
    from tqdm import tqdm

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou = 0.0
    best_state = None
    warmup_epochs = 5

    for epoch in range(epochs):
        # Unfreeze encoder after warmup
        if epoch == warmup_epochs and hasattr(model, 'unfreeze_encoder'):
            print(f"Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

        # Train
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        if stage == 1:
            intersection, union = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device).long()
                    pred = model.predict_mask(images)
                    intersection += (pred * masks).sum().item()
                    union += (pred + masks).clamp(0, 1).sum().item()
            iou = intersection / max(union, 1)
        else:
            tp, fp, fn = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device).long()
                    corrosion_mask = (masks == 2).long()
                    pred = model.predict_mask(images)
                    pred_c = (pred == 2).long()
                    tp += (pred_c * corrosion_mask).sum().item()
                    fp += (pred_c * (1 - corrosion_mask)).sum().item()
                    fn += ((1 - pred_c) * corrosion_mask).sum().item()
            iou = tp / max(tp + fp + fn, 1)

        scheduler.step()

        if iou > best_iou:
            best_iou = iou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, iou={iou:.4f}, best_iou={best_iou:.4f}")

    # ── Export best model to TorchScript ─────────────────────
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    model_path = Path(f"/app/models/stage{stage}_{stage_name}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # For HuggingFace models, we need to trace the inner model
    inner_model = model.model if hasattr(model, 'model') else model
    dummy = torch.randn(1, 3, 512, 512).to(device)

    try:
        scripted = torch.jit.trace(inner_model, dummy)
        scripted.save(str(model_path))
        size_mb = model_path.stat().st_size / 1e6
        print(f"Exported TorchScript model: {model_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        # Save state dict as fallback
        torch.save(best_state, str(model_path).replace(".pt", "_state_dict.pt"))
        model_path = Path(str(model_path).replace(".pt", "_state_dict.pt"))

    # ── Upload to S3 ─────────────────────────────────────────
    s3_uri = ""
    if upload_artifacts and s3_output_path:
        s3_uri = f"{s3_output_path}stage{stage}_{stage_name}.pt"
        print(f"Uploading model to {s3_uri}...")
        os.system(f"aws s3 cp {model_path} {s3_uri}")
        print(f"Upload complete")

    # ── Go/no-go gate for Stage 2 ────────────────────────────
    gate_status = ""
    if stage == 2:
        if best_iou >= 0.45:
            gate_status = "GO"
        elif best_iou >= 0.25:
            gate_status = "CONDITIONAL"
        else:
            gate_status = "NO-GO"

    result = {
        "stage": stage,
        "stage_name": stage_name,
        "best_iou": round(best_iou, 4),
        "epochs_trained": epochs,
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 1) if model_path.exists() else 0,
        "s3_uri": s3_uri,
        "param_count_m": round(param_count, 1),
        "device": str(device),
    }
    if gate_status:
        result["gate_status"] = gate_status

    return result


# ═══════════════════════════════════════════════════════════════
# Handler
# ═══════════════════════════════════════════════════════════════

def handler(job: dict) -> dict:
    """RunPod Serverless handler for model training."""
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")

    stage = job_input.get("stage", 1)
    print(f"Starting Stage {stage} training job: {job_id}")

    try:
        result = run_training(job_input)
        print(f"Training complete: {result}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "stage": stage}


runpod.serverless.start({"handler": handler})
