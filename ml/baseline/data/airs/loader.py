"""AIRS (Aerial Imagery for Roof Segmentation) dataset loader.

Source: https://github.com/yanglikai/AIRS
Labels: Pixel-level roof masks (binary: roof / non-roof)
GSD: 7.5 cm, Christchurch NZ

License: Research-only ⚠️ — check LINZ/Canterbury terms for commercial use.
"""

from pathlib import Path

import numpy as np
import rasterio
from torch.utils.data import Dataset


class AIRSRoofDataset(Dataset):
    """PyTorch dataset for AIRS roof footprint segmentation.

    Returns:
        - image: (3, H, W) float32 RGB
        - mask: (H, W) int64 — 0=background, 1=roof
        - metadata: dict with tile_id, gsd
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
            "metadata": {"tile_id": tile_id, "gsd": gsd},
        }
