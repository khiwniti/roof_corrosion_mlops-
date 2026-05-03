#!/usr/bin/env bash
# Download open datasets for Phase 1a baseline.
# Each dataset is downloaded to data/{dataset_name}/
#
# Usage:
#   bash ml/baseline/data/download_datasets.sh [caribbean|airs|spacenet|xbd|all]
#
# Prerequisites:
#   - awscli (for SpaceNet on S3)
#   - wget/curl

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR"

download_caribbean() {
    echo "══════════════════════════════════════════════════════"
    echo "  Open AI Caribbean Challenge"
    echo "  License: CC-BY-4.0 ✅ Commercial OK"
    echo "══════════════════════════════════════════════════════"

    local dest="$DATA_DIR/caribbean"
    mkdir -p "$dest"

    # The Caribbean dataset is available via Radiant MLHub / Drivendata
    # Option 1: DrivenData direct download (requires free account)
    echo "Download from: https://www.drivendata.org/competitions/60/building-separation/"
    echo ""
    echo "Steps:"
    echo "  1. Register at https://www.drivendata.org/"
    echo "  2. Accept competition terms"
    echo "  3. Download tarball from competition data page"
    echo "  4. Extract to $dest/"
    echo ""
    echo "Expected structure:"
    echo "  $dest/"
    echo "  ├── train/"
    echo "  │   ├── images/     (.tif tiles)"
    echo "  │   └── labels.geojson"
    echo "  └── val/"
    echo "      ├── images/"
    echo "      └── labels.geojson"

    # Option 2: Radiant MLHub (API key required)
    # pip install radiant-mlhub
    # mlhub authenticate --api_key YOUR_KEY
    # mlhub download ref_african_crops_uganda_01 "$dest"

    if [ ! -f "$dest/.downloaded" ]; then
        echo ""
        echo "⚠️  Manual download required. Create $dest/.downloaded when done."
    else
        echo "✅ Caribbean dataset already downloaded."
    fi
}

download_airs() {
    echo "══════════════════════════════════════════════════════"
    echo "  AIRS — Aerial Imagery for Roof Segmentation"
    echo "  License: Research-only ⚠️ Check terms for commercial"
    echo "══════════════════════════════════════════════════════"

    local dest="$DATA_DIR/airs"
    mkdir -p "$dest"

    echo "Download from: https://github.com/yanglikai/AIRS"
    echo ""
    echo "Steps:"
    echo "  1. Clone or download from GitHub"
    echo "  2. Extract to $dest/"
    echo ""
    echo "Expected structure:"
    echo "  $dest/"
    echo "  ├── train/"
    echo "  │   ├── images/     (.tif tiles, 7.5cm GSD)"
    echo "  │   └── masks/      (.tif binary masks)"
    echo "  └── val/"
    echo "      ├── images/"
    echo "      └── masks/"

    if [ ! -f "$dest/.downloaded" ]; then
        echo ""
        echo "⚠️  Manual download required. Create $dest/.downloaded when done."
    else
        echo "✅ AIRS dataset already downloaded."
    fi
}

download_spacenet() {
    echo "══════════════════════════════════════════════════════"
    echo "  SpaceNet 7 (Maxar 30cm, building footprints)"
    echo "  License: CC-BY-SA-4.0 ✅ Commercial OK (attribution)"
    echo "══════════════════════════════════════════════════════"

    local dest="$DATA_DIR/spacenet"
    mkdir -p "$dest"

    # SpaceNet is hosted on AWS S3 public bucket
    echo "Downloading from AWS S3 (s3://spacenet-dataset/)..."

    if command -v aws &>/dev/null; then
        echo "  Using awscli to download SpaceNet 7 (urban extraction)..."
        echo ""
        echo "  # List available datasets:"
        echo "  aws s3 ls s3://spacenet-dataset/spacenet/SN7_buildings/ --no-sign-request"
        echo ""
        echo "  # Download train split:"
        echo "  aws s3 sync s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz \"$dest/\" --no-sign-request"
        echo ""
        echo "  # Or download specific AOIs:"
        echo "  aws s3 sync s3://spacenet-dataset/spacenet/SN7_buildings/train/ \"$dest/train/\" --no-sign-request"
    else
        echo "⚠️  awscli not found. Install with: pip install awscli"
        echo "  Then run: aws s3 sync s3://spacenet-dataset/spacenet/SN7_buildings/train/ $dest/train/ --no-sign-request"
    fi

    echo ""
    echo "Expected structure:"
    echo "  $dest/"
    echo "  ├── train/"
    echo "  │   ├── images/     (.tif Maxar tiles)"
    echo "  │   └── labels.geojson"
    echo "  └── val/"
    echo "      ├── images/"
    echo "      └── labels.geojson"
}

download_xbd() {
    echo "══════════════════════════════════════════════════════"
    echo "  xBD (xView2) — Building Damage on Maxar"
    echo "  License: CC-BY-NC-4.0 ❌ NON-COMMERCIAL"
    echo "  ⚠️  RESEARCH-ONLY — do NOT use in production models"
    echo "══════════════════════════════════════════════════════"

    local dest="$DATA_DIR/xbd"
    mkdir -p "$dest"

    echo "Download from: https://xview2.org/dataset"
    echo ""
    echo "Steps:"
    echo "  1. Register at https://xview2.org/"
    echo "  2. Accept NC license terms"
    echo "  3. Download train/test tarballs"
    echo "  4. Extract to $dest/"
    echo ""
    echo "⚠️  REMINDER: This dataset is CC-BY-NC-4.0."
    echo "   Models pretrained on xBD must be fine-tuned on"
    echo "   commercially-safe data before production use."
    echo "   The CI license gate will block any xBD-tagged run"
    echo "   from promotion to staging/production."

    if [ ! -f "$dest/.downloaded" ]; then
        echo ""
        echo "⚠️  Manual download required. Create $dest/.downloaded when done."
    else
        echo "✅ xBD dataset already downloaded."
    fi
}

# ── Main ──────────────────────────────────────────────────
DATASET="${1:-all}"

case "$DATASET" in
    caribbean) download_caribbean ;;
    airs)      download_airs ;;
    spacenet)  download_spacenet ;;
    xbd)       download_xbd ;;
    all)
        download_caribbean
        echo ""
        download_airs
        echo ""
        download_spacenet
        echo ""
        download_xbd
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Usage: $0 [caribbean|airs|spacenet|xbd|all]"
        exit 1
        ;;
esac

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Next steps after downloading:"
echo "  1. Verify structure: ls -R $DATA_DIR/<dataset>/"
echo "  2. Run license check: python ml/data/check_licenses.py"
echo "  3. Start training: python ml/baseline/train_stage2_corrosion.py"
echo "══════════════════════════════════════════════════════"
