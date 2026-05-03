"""xBD (xView2) dataset loader.

Source: https://xview2.org/
Labels: Building damage (no-damage, minor, major, destroyed) on Maxar 30–80cm
License: CC-BY-NC-4.0 ❌ NON-COMMERCIAL

⚠️ RESEARCH-ONLY — weights trained on xBD MUST NOT be promoted to production.
    Use for pretraining only, then fine-tune on commercially-safe data.
"""

from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset

XBD_DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

# License tag for MLflow enforcement
DATA_LICENSE_TAG = "CC-BY-NC-4.0"
COMMERCIAL_SAFE = False


class xBDDamageDataset(Dataset):
    """PyTorch dataset for xBD building damage classification/segmentation.

    ⚠️ RESEARCH-ONLY — do not use in production model training pipelines.

    Returns:
        - image: (3, H, W) float32 RGB (post-disaster)
        - mask: (H, W) int64 — 0=background, 1–4 damage levels
        - metadata: dict with tile_id, gsd, license=CC-BY-NC-4.0
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
        crop_size: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.crop_size = crop_size

        self.tiles = sorted(self.image_dir.glob("*.tif"))
        if not self.tiles:
            raise FileNotFoundError(f"No .tif tiles found in {image_dir}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile_path = self.tiles[idx]
        tile_id = tile_path.stem

        with rasterio.open(tile_path) as src:
            img = src.read([1, 2, 3]).astype(np.float32) / 255.0
            gsd = src.res[0]

        mask_path = self.mask_dir / f"{tile_id}.tif"
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64)
        else:
            mask = np.zeros(img.shape[1:], dtype=np.int64)

        if self.transform:
            transformed = self.transform(
                image=np.transpose(img, (1, 2, 0)), mask=mask
            )
            img = np.transpose(transformed["image"], (2, 0, 1))
            mask = transformed["mask"]

        import torch

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "metadata": {
                "tile_id": tile_id,
                "gsd": gsd,
                "license": DATA_LICENSE_TAG,
                "commercial_safe": COMMERCIAL_SAFE,
            },
        }
