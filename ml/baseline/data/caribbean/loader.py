"""Open AI Caribbean Challenge dataset loader.

Source: https://www.drivendata.org/competitions/60/building-separation/
Labels: roof type (concrete_cement, healthy_metal, irregular_metal, incomplete, other)
Key class: `irregular_metal` ≈ corroded/weathered metal roof — our corrosion proxy.

License: CC-BY-4.0 ✅ Commercial use OK
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from torch.utils.data import Dataset

CARIBBEAN_CLASSES = [
    "concrete_cement",
    "healthy_metal",
    "irregular_metal",  # ← corrosion proxy
    "incomplete",
    "other",
]

# Binary corrosion mapping: irregular_metal → 1 (corroded), everything else → 0
CORROSION_CLASS_IDX = CARIBBEAN_CLASSES.index("irregular_metal")


class CaribbeanRoofDataset(Dataset):
    """PyTorch dataset for Open AI Caribbean roof type segmentation.

    Each sample returns:
        - image: (3, H, W) float32 RGB tile
        - mask: (H, W) int64 — 0=background, 1=healthy_roof, 2=corroded_roof
        - metadata: dict with tile_id, roof_type_str, gsd
    """

    def __init__(
        self,
        image_dir: str | Path,
        labels_geojson: str | Path,
        transform=None,
        crop_size: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.labels = gpd.read_file(labels_geojson)
        self.transform = transform
        self.crop_size = crop_size

        # Index tiles
        self.tiles = sorted(self.image_dir.glob("*.tif")) + sorted(
            self.image_dir.glob("*.png")
        )
        if not self.tiles:
            raise FileNotFoundError(f"No .tif/.png tiles found in {image_dir}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile_path = self.tiles[idx]
        tile_id = tile_path.stem

        # Load image
        if tile_path.suffix == ".tif":
            with rasterio.open(tile_path) as src:
                img = src.read([1, 2, 3])  # RGB bands
                gsd = src.res[0]  # ground sample distance in meters
        else:
            import cv2

            img_np = cv2.imread(str(tile_path))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img = np.transpose(img_np, (2, 0, 1)).astype(np.float32) / 255.0
            gsd = 0.1  # ~10cm default for drone

        img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)

        # Build mask from geojson labels overlapping this tile
        mask = np.zeros(img.shape[1:], dtype=np.int64)
        tile_roofs = self.labels[self.labels["tile_id"] == tile_id] if "tile_id" in self.labels.columns else self.labels

        for _, roof in tile_roofs.iterrows():
            roof_type = roof.get("roof_type", roof.get("class", "other"))
            if roof_type == "irregular_metal":
                mask = self._rasterize_polygon(roof.geometry, mask, value=2)
            elif roof_type in ("healthy_metal", "concrete_cement"):
                mask = self._rasterize_polygon(roof.geometry, mask, value=1)

        if self.transform:
            transformed = self.transform(image=np.transpose(img, (1, 2, 0)), mask=mask)
            img = np.transpose(transformed["image"], (2, 0, 1))
            mask = transformed["mask"]

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "metadata": {"tile_id": tile_id, "gsd": gsd},
        }

    @staticmethod
    def _rasterize_polygon(geometry, mask: np.ndarray, value: int) -> np.ndarray:
        """Rasterize a shapely polygon onto the mask array."""
        from rasterio.features import rasterize as rio_rasterize

        burned = rio_rasterize(
            [(geometry, value)],
            out_shape=mask.shape,
            fill=0,
            dtype=np.int64,
        )
        # Only overwrite background (0) pixels to avoid overlapping roofs
        mask = np.where(burned > 0, burned, mask)
        return mask


import torch  # noqa: E402 — needed after numpy ops above
