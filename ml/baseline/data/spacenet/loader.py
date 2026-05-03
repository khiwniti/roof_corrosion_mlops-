"""SpaceNet dataset loader (buildings 2–7 variants).

Source: https://spacenet.ai/datasets/
Labels: Building polygon footprints on Maxar 30–50cm imagery
License: CC-BY-SA-4.0 ✅ Commercial use OK (with attribution)

Best generic pretraining for Maxar-domain roof segmentation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
from torch.utils.data import Dataset


class SpaceNetBuildingDataset(Dataset):
    """PyTorch dataset for SpaceNet building footprint segmentation.

    Returns:
        - image: (3, H, W) float32 RGB
        - mask: (H, W) int64 — 0=background, 1=building
        - metadata: dict with tile_id, gsd
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
            transform = src.transform
            crs = src.crs

        # Rasterize building polygons
        mask = np.zeros(img.shape[1:], dtype=np.int64)
        tile_buildings = self.labels[self.labels.intersects(
            self._tile_bounds(tile_path)
        )]

        if len(tile_buildings) > 0:
            shapes = [(geom, 1) for geom in tile_buildings.geometry]
            mask = rio_rasterize(shapes, out_shape=mask.shape, transform=transform, fill=0, dtype=np.int64)

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

    @staticmethod
    def _tile_bounds(tile_path: Path):
        """Get bounding box of a GeoTIFF tile."""
        with rasterio.open(tile_path) as src:
            return src.bounds
