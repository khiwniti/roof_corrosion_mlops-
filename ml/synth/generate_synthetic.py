"""Synthetic roof corrosion dataset generator.

Generates realistic-looking overhead rooftop images with parametric
corrosion patterns for pipeline validation and quick training tests.

This is NOT a replacement for real data — it's for:
1. Validating the training pipeline runs end-to-end
2. Quick sanity checks before committing to long GPU training
3. Unit tests that need deterministic input

Usage:
    python ml/synth/generate_synthetic.py --num-tiles 200 --output-dir data/synthetic/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def generate_roof_tile(
    size: int = 512,
    roof_type: str = "metal",
    corrosion_level: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single synthetic overhead roof tile with optional corrosion.

    Args:
        size: tile size in pixels
        roof_type: 'metal', 'tile', or 'asphalt'
        corrosion_level: 0.0 (none) to 1.0 (severe)

    Returns:
        (image, mask) — (H, W, 3) uint8 image and (H, W) int64 mask
        Mask classes: 0=background, 1=roof, 2=corrosion
    """
    rng = np.random.RandomState(seed)

    # ── Background (ground/vegetation) ───────────────────────
    bg_color = rng.randint([60, 80, 40], [100, 130, 80])
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = bg_color

    # Add noise to background
    noise = rng.randint(-15, 15, (size, size, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── Roof polygon ─────────────────────────────────────────
    mask = np.zeros((size, size), dtype=np.uint8)  # uint8 for OpenCV compat

    # Random roof shape (rectangle with slight rotation)
    cx, cy = size // 2 + rng.randint(-50, 50), size // 2 + rng.randint(-50, 50)
    w = rng.randint(size // 4, size // 2)
    h = rng.randint(size // 4, size // 2)
    angle = rng.randint(-30, 30)

    # Create rotated rectangle
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Roof color based on type
    if roof_type == "metal":
        roof_base = rng.randint([140, 150, 155], [180, 190, 200])  # silver/gray
    elif roof_type == "tile":
        roof_base = rng.randint([140, 70, 40], [190, 110, 70])  # terracotta
    else:  # asphalt
        roof_base = rng.randint([50, 50, 55], [80, 80, 85])  # dark gray

    # Fill roof
    cv2.fillPoly(image, [box], roof_base.tolist())
    cv2.fillPoly(mask, [box], 1)

    # Add roof texture (ridges for metal, grid for tile)
    if roof_type == "metal":
        # Horizontal ridges
        for y in range(box[:, 1].min(), box[:, 1].max(), rng.randint(8, 15)):
            ridge_color = np.clip(roof_base + rng.randint(-10, 10), 0, 255).astype(np.uint8)
            cv2.line(image, (box[:, 0].min(), y), (box[:, 0].max(), y), ridge_color.tolist(), 1)
    elif roof_type == "tile":
        # Grid pattern
        for y in range(box[:, 1].min(), box[:, 1].max(), rng.randint(15, 25)):
            for x in range(box[:, 0].min(), box[:, 0].max(), rng.randint(15, 25)):
                tile_color = np.clip(roof_base + rng.randint(-20, 20), 0, 255).astype(np.uint8)
                cv2.rectangle(image, (x, y), (x + 12, y + 12), tile_color.tolist(), 1)

    # ── Corrosion patches ────────────────────────────────────
    if corrosion_level > 0:
        num_patches = max(1, int(corrosion_level * rng.randint(3, 12)))
        corrosion_mask = np.zeros((size, size), dtype=np.uint8)

        for _ in range(num_patches):
            # Random position within roof
            px = cx + rng.randint(-w // 3, w // 3)
            py = cy + rng.randint(-h // 3, h // 3)
            patch_size = rng.randint(15, int(50 * corrosion_level + 20))

            # Corrosion color: orange-brown rust
            rust_color = rng.randint([120, 60, 20], [200, 120, 50])

            # Draw irregular corrosion patch
            pts = []
            n_pts = rng.randint(5, 10)
            for _ in range(n_pts):
                angle_pt = rng.uniform(0, 2 * np.pi)
                r = rng.uniform(0.3, 1.0) * patch_size
                pts.append([int(px + r * np.cos(angle_pt)), int(py + r * np.sin(angle_pt))])
            pts = np.array(pts, dtype=np.int32)

            # Only apply within roof area
            corrosion_roi = np.zeros_like(corrosion_mask)
            cv2.fillPoly(corrosion_roi, [pts], 1)
            corrosion_roi = corrosion_roi & (mask == 1).astype(np.uint8)

            if corrosion_roi.sum() > 0:
                cv2.fillPoly(image, [pts], rust_color.tolist())
                corrosion_mask |= corrosion_roi

        # Add corrosion texture (mottling)
        corrosion_pixels = np.where(corrosion_mask > 0)
        if len(corrosion_pixels[0]) > 0:
            noise = rng.randint(-20, 20, (len(corrosion_pixels[0]), 3), dtype=np.int16)
            image[corrosion_pixels[0], corrosion_pixels[1]] = np.clip(
                image[corrosion_pixels[0], corrosion_pixels[1]].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

            # Update mask: corrosion = class 2
            mask[corrosion_mask > 0] = 2

    # ── Add shadows and lighting ─────────────────────────────
    shadow_angle = rng.uniform(0, 2 * np.pi)
    shadow_offset = int(rng.randint(10, 30))
    shadow_shift = (int(shadow_offset * np.cos(shadow_angle)), int(shadow_offset * np.sin(shadow_angle)))

    # Simple shadow: darken area offset from roof
    shadow_mask = np.zeros((size, size), dtype=np.uint8)
    M = np.float32([[1, 0, shadow_shift[0]], [0, 1, shadow_shift[1]]])
    shifted_roof = ((mask > 0).astype(np.uint8) * 255)
    shadow_mask = cv2.warpAffine(shifted_roof, M, (size, size))
    shadow_mask = shadow_mask & (mask == 0).astype(np.uint8)  # only on background

    shadow_pixels = np.where(shadow_mask > 0)
    if len(shadow_pixels[0]) > 0:
        image[shadow_pixels[0], shadow_pixels[1]] = np.clip(
            image[shadow_pixels[0], shadow_pixels[1]].astype(np.int16) - 30, 0, 255
        ).astype(np.uint8)

    # ── JPEG compression artifacts (simulates satellite) ────
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(85, 100)]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return image, mask.astype(np.int64)


def generate_dataset(
    num_tiles: int = 200,
    output_dir: str = "data/synthetic",
    tile_size: int = 512,
    train_split: float = 0.8,
):
    """Generate a full synthetic dataset with train/val split."""
    output = Path(output_dir)
    rng = np.random.RandomState(42)

    # Severity distribution (weighted toward healthy roofs — realistic)
    corrosion_levels = [
        (0.0, 0.40),     # 40% no corrosion
        (0.05, 0.20),    # 20% light
        (0.15, 0.20),    # 20% moderate
        (0.40, 0.15),    # 15% significant
        (0.70, 0.05),    # 5% severe
    ]

    roof_types = ["metal", "tile", "asphalt"]
    roof_weights = [0.5, 0.3, 0.2]

    metadata = []

    for i in tqdm(range(num_tiles), desc="Generating tiles"):
        # Sample corrosion level
        r = rng.random()
        cumulative = 0
        corrosion = 0.0
        for level, weight in corrosion_levels:
            cumulative += weight
            if r <= cumulative:
                corrosion = level
                break

        roof_type = rng.choice(roof_types, p=roof_weights)

        image, mask = generate_roof_tile(
            size=tile_size,
            roof_type=roof_type,
            corrosion_level=corrosion,
            seed=i,
        )

        # Determine split
        split = "train" if i < int(num_tiles * train_split) else "val"

        # Save
        tile_id = f"synth_{i:05d}"
        img_dir = output / split / "images"
        mask_dir = output / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_dir / f"{tile_id}.png"), image)
        cv2.imwrite(str(mask_dir / f"{tile_id}.png"), mask)

        # Compute stats for metadata
        roof_pixels = (mask == 1).sum() + (mask == 2).sum()
        corrosion_pixels = (mask == 2).sum()
        corrosion_pct = (corrosion_pixels / max(roof_pixels, 1)) * 100

        metadata.append({
            "tile_id": tile_id,
            "split": split,
            "roof_type": roof_type,
            "corrosion_level": corrosion,
            "corrosion_percent": round(corrosion_pct, 1),
            "roof_pixels": int(roof_pixels),
            "corrosion_pixels": int(corrosion_pixels),
        })

    # Save metadata
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    train_count = sum(1 for m in metadata if m["split"] == "train")
    val_count = sum(1 for m in metadata if m["split"] == "val")
    corroded = sum(1 for m in metadata if m["corrosion_percent"] > 5)

    print(f"\n{'='*60}")
    print(f"Synthetic dataset generated: {output}")
    print(f"  Train: {train_count} tiles")
    print(f"  Val:   {val_count} tiles")
    print(f"  Corroded (>5%): {corroded} tiles ({corroded/num_tiles*100:.0f}%)")
    print(f"  Healthy: {num_tiles - corroded} tiles ({(num_tiles-corroded)/num_tiles*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic roof corrosion dataset")
    parser.add_argument("--num-tiles", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--train-split", type=float, default=0.8)
    args = parser.parse_args()
    generate_dataset(args.num_tiles, args.output_dir, args.tile_size, args.train_split)
