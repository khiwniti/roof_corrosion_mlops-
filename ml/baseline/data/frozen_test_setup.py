"""Frozen test set setup script.

Creates the directory structure for the immutable frozen real-image test set.
This set is NEVER used in training and NEVER regenerated.

To populate with real Maxar/Nearmap preview tiles:
1. Get trial credits from Maxar (SecureWatch) or Nearmap (MapBrowser API)
2. Download ~100–200 tiles over target regions (industrial areas with known corroded roofs)
3. Hand-label in Label Studio with classes: background, roof, corrosion
4. Export masks to data/frozen_test/masks/
5. Lock with DVC: dvc add data/frozen_test && git add data/frozen_test.dvc

Usage:
    python ml/baseline/data/frozen_test_setup.py [--num-tiles 200]
"""

import argparse
from pathlib import Path

FROZEN_DIR = Path("data/frozen_test")


def setup_frozen_test(num_tiles: int = 200):
    """Create frozen test set directory structure and metadata."""

    dirs = [
        FROZEN_DIR / "images",
        FROZEN_DIR / "masks",
        FROZEN_DIR / "metadata",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Write README
    readme = FROZEN_DIR / "README.md"
    readme.write_text(f"""# Frozen Real-Image Test Set

## ⚠️ IMMUTABLE — DO NOT MODIFY

This test set is the ground truth for all model evaluation.
It must NEVER be used in training, and must NEVER be regenerated.

- **Target size**: ~{num_tiles} tiles
- **Source**: Maxar SecureWatch / Nearmap MapBrowser preview tiles
- **GSD**: 30–50 cm (Maxar) or 7–10 cm (Nearmap)
- **Regions**: Industrial areas with known corroded metal roofs
- **Labels**: 3 classes — background (0), roof (1), corrosion (2)
- **Labeling tool**: Label Studio (self-hosted)
- **Versioning**: DVC-locked, checksummed in CI

## Directory structure

```
frozen_test/
├── images/          # GeoTIFF tiles (RGB)
├── masks/           # GeoTIFF masks (int: 0/1/2)
├── metadata/        # Per-tile metadata (capture date, GSD, region)
└── manifest.json    # List of tile IDs + checksums
```

## Populating the test set

1. **Get trial credits**:
   - Maxar: https://www.maxar.com/secure-watch (free trial)
   - Nearmap: https://www.nearmap.com/au/mapbrowser-api (free trial)

2. **Select regions** with known corroded metal roofs:
   - Industrial parks in Southeast Asia (Indonesia, Vietnam, Philippines)
   - Shipping/container yards
   - Agricultural buildings with corrugated metal

3. **Download tiles** via the tile fetch API:
   ```python
   from services.api.app.inference.tile_fetch import fetch_tile
   tile = fetch_tile(lat=-6.2, lng=106.8, zoom=20, source="maxar")
   ```

4. **Label in Label Studio**:
   - Start Label Studio: `label-studio start`
   - Create project with segmentation template
   - Import tiles, label with 3 classes
   - Export masks to `masks/`

5. **Lock with DVC**:
   ```bash
   dvc add data/frozen_test
   git add data/frozen_test.dvc data/frozen_test/.gitignore
   git commit -m "lock frozen test set v1"
   ```

## CI enforcement

The eval harness (`ml/baseline/eval_frozen.py`) reads from this directory.
CI validates that the DVC checksum matches the committed `.dvc` file.
If checksums differ, CI fails — preventing silent mutation of the test set.
""")

    # Write manifest template
    manifest = FROZEN_DIR / "metadata" / "manifest_template.json"
    manifest.write_text("""{
  "version": "1.0.0",
  "created_at": "TBD",
  "num_tiles": 0,
  "source": "maxar_or_nearmap",
  "regions": [],
  "tiles": [
    {
      "tile_id": "example_tile_001",
      "image_path": "images/example_tile_001.tif",
      "mask_path": "masks/example_tile_001.tif",
      "gsd_m": 0.3,
      "capture_date": "2024-01-15",
      "region": "Jakarta_Industrial",
      "sha256": "TBD"
    }
  ],
  "label_classes": {
    "0": "background",
    "1": "roof",
    "2": "corrosion"
  }
}
""")

    # Write .gitkeep files
    for d in dirs:
        (d / ".gitkeep").touch()

    print(f"Frozen test set directory created at {FROZEN_DIR}/")
    print(f"  Images:  {FROZEN_DIR / 'images/'}")
    print(f"  Masks:   {FROZEN_DIR / 'masks/'}")
    print(f"  Meta:    {FROZEN_DIR / 'metadata/'}")
    print(f"\nNext: Populate with {num_tiles} Maxar/Nearmap preview tiles + hand-label.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup frozen test set directory")
    parser.add_argument("--num-tiles", type=int, default=200)
    args = parser.parse_args()
    setup_frozen_test(args.num_tiles)
