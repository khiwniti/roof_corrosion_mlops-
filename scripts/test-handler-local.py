#!/usr/bin/env python3
"""Local development script to test the RunPod handler without external APIs.

Runs both Tier-0 and Tier-1 pipelines with stub fallbacks, printing
results and saving masks to a local directory.

Usage:
    python scripts/test-handler-local.py

Environment:
    PYTHONPATH must include ml/ and infra/runpod/serverless/tiered/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project paths for local imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml"))
sys.path.insert(0, str(REPO_ROOT / "infra/runpod/serverless/tiered"))

from handler import handler


TEST_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [100.5018, 13.7563],
            [100.5028, 13.7563],
            [100.5028, 13.7573],
            [100.5018, 13.7573],
            [100.5018, 13.7563],
        ]
    ],
}


def run_tier(tier: int) -> dict:
    """Run handler for a given tier and return result."""
    print(f"\n{'=' * 60}")
    print(f"  Testing Tier {tier}")
    print(f"{'=' * 60}")

    job = {
        "input": {
            "polygon": TEST_POLYGON,
            "tier": tier,
            "jobId": f"local-test-tier{tier}",
            "webhook": "",  # No webhook in local test
        }
    }

    result = handler(job)
    return result


def save_result(result: dict, tier: int, out_dir: Path) -> None:
    """Save JSON result and mask image to output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / f"tier{tier}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved result: {json_path}")

    # Save mask if present
    mask_url = result.get("mask_url", "")
    if mask_url.startswith("data:image/png;base64,"):
        import base64

        b64 = mask_url[len("data:image/png;base64,") :]
        png_path = out_dir / f"tier{tier}_mask.png"
        with open(png_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  Saved mask:   {png_path}")


def print_summary(result: dict, tier: int) -> None:
    """Print human-readable summary of handler result."""
    print(f"\n  Status: {result.get('status', 'unknown')}")
    print(f"  Job ID: {result.get('jobId', 'N/A')}")
    print(f"  Processing: {result.get('processing_ms', 'N/A')} ms")

    area = result.get("area", {})
    if area:
        print(f"  Roof Area: {area.get('roof_area_m2', 'N/A')} m²")
        print(f"  Buildings: {area.get('building_count', 'N/A')}")

    pred = result.get("prediction", {})
    if pred:
        print(f"  Model: {pred.get('model_version', 'N/A')}")
        print(f"  Confidence: {pred.get('confidence', 'N/A')}")
        print(f"  Corrosion: {pred.get('corrosion_prob', 'N/A')}")
        print(f"  Severity: {pred.get('severity', 'N/A')}")
        coarse = pred.get("coarse_breakdown", {})
        if coarse:
            print(f"  Materials: metal={coarse.get('metal_percent', 'N/A')}%, tile={coarse.get('tile_percent', 'N/A')}%")

    ingest = result.get("ingestion_meta", {})
    if ingest:
        print(f"  Ingestion: {ingest.get('ingestion_method', 'N/A')}")


def main() -> None:
    out_dir = Path("local_test_outputs")

    # Tier 0
    result0 = run_tier(0)
    print_summary(result0, 0)
    save_result(result0, 0, out_dir)

    # Tier 1
    result1 = run_tier(1)
    print_summary(result1, 1)
    save_result(result1, 1, out_dir)

    print(f"\n{'=' * 60}")
    print("  Local test complete")
    print(f"  Outputs: {out_dir.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
