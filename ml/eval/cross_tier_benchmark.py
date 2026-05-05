"""Cross-tier evaluation benchmark.

Compares Tier-0 (S2), Tier-1 (VHR), and Tier-3 (drone) predictions
on the same polygon to measure improvement between tiers.

Architecture: ADR-001, SPEC-phase-3

Usage:
    python ml/eval/cross_tier_benchmark.py --polygon polygon.json --output report.md
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("eval.cross_tier_benchmark")


def run_tier0(polygon: dict) -> dict[str, Any]:
    """Run Tier-0 S2 pipeline and return metrics."""
    logger.info("Running Tier-0 benchmark")

    # Stub: random features
    features = np.random.rand(512, 512, 10).astype(np.float32)
    feature_names = ["ndvi", "B11", "B12", "B08", "B04", "iron_oxide", "B03", "B02", "B05", "B06"]

    from inference.tier0 import predict, estimate_roof_area

    pred = predict(features, feature_names)
    area = estimate_roof_area(pred["material_mask"], gsd_m=10.0, building_count=5)

    return {
        "tier": 0,
        "gsd_m": 10.0,
        "material_mIoU": 0.45,  # stub benchmark
        "corrosion_mIoU": 0.30,
        "severity_accuracy": 0.55,
        "roof_area_m2": area["roof_area_m2"],
        "building_count": area["building_count"],
        "material_breakdown": pred["coarse_breakdown"],
        "corrosion_prob": pred["corrosion_prob"],
        "severity": pred["severity"],
        "confidence": pred["confidence"],
        "model_version": pred["model_version"],
    }


def run_tier1(polygon: dict) -> dict[str, Any]:
    """Run Tier-1 VHR pipeline and return metrics."""
    logger.info("Running Tier-1 benchmark")

    from inference.tier1 import predict

    # Synthetic VHR image
    vhr = np.random.rand(1024, 1024, 3).astype(np.float32)
    pred = predict(vhr, model=None, temperature=1.0)

    # Estimate area at 0.5 m GSD
    pixel_area = 0.5 * 0.5
    roof_pixels = pred["total_roof_area_px"]
    area_m2 = roof_pixels * pixel_area

    # Derive corrosion and severity from building results
    buildings = pred["buildings"]
    corrosion_prob = max((r["corrosion_prob"] for r in buildings), default=0.0) if buildings else 0.0
    severity_probs_list = [r["severity_probs"] for r in buildings] if buildings else [[1, 0, 0, 0]]
    avg_severity_probs = np.mean(severity_probs_list, axis=0) if severity_probs_list else [1, 0, 0, 0]
    severity_idx = int(np.argmax(avg_severity_probs))
    severity_labels = ["none", "light", "moderate", "severe"]
    severity = severity_labels[severity_idx]

    return {
        "tier": 1,
        "gsd_m": 0.5,
        "material_mIoU": 0.65,
        "corrosion_mIoU": 0.50,
        "severity_accuracy": 0.72,
        "roof_area_m2": round(area_m2, 1),
        "building_count": pred["building_count"],
        "material_breakdown": pred["material_breakdown"],
        "corrosion_prob": float(corrosion_prob),
        "severity": severity,
        "confidence": float(pred["confidence"]),
        "model_version": pred["model_version"],
    }


def run_tier3(polygon: dict) -> dict[str, Any]:
    """Run Tier-3 drone pipeline and return metrics."""
    logger.info("Running Tier-3 benchmark")

    # Tier-3 is ground truth + estimator HITL
    # Stub: assume perfect drone imagery with manual trace
    return {
        "tier": 3,
        "gsd_m": 0.05,
        "material_mIoU": 0.85,
        "corrosion_mIoU": 0.75,
        "severity_accuracy": 0.90,
        "roof_area_m2": 420.0,
        "building_count": 1,
        "material_breakdown": {"metal": 80.0, "tile": 15.0, "other": 5.0},
        "corrosion_prob": 0.35,
        "severity": "moderate",
        "confidence": 0.95,
        "model_version": "drone-hitl-v1",
        "note": "Ground truth from estimator manual trace on drone imagery",
    }


def compute_delta(baseline: dict, target: dict) -> dict[str, Any]:
    """Compute delta metrics between two tiers."""
    deltas = {}
    for key in ["material_mIoU", "corrosion_mIoU", "severity_accuracy", "confidence"]:
        b = baseline.get(key, 0)
        t = target.get(key, 0)
        deltas[key] = {
            "absolute": round(t - b, 4),
            "relative": round((t - b) / b * 100, 1) if b > 0 else None,
        }
    return deltas


def generate_report(
    polygon: dict,
    tier0: dict,
    tier1: dict,
    tier3: dict,
) -> str:
    """Generate markdown benchmark report."""
    t0_to_t1 = compute_delta(tier0, tier1)
    t1_to_t3 = compute_delta(tier1, tier3)

    lines = [
        "# Cross-Tier Evaluation Benchmark",
        "",
        f"**Date:** {datetime.now(UTC).isoformat()}  ",
        f"**Polygon:** {json.dumps(polygon)[:200]}...",
        "",
        "## Tier Summary",
        "",
        "| Tier | GSD | Material mIoU | Corrosion mIoU | Severity Acc | Confidence | Area (m²) |",
        "|------|-----|---------------|----------------|--------------|------------|-----------|",
        f"| Tier-0 (S2) | {tier0['gsd_m']} m | {tier0['material_mIoU']:.2f} | {tier0['corrosion_mIoU']:.2f} | {tier0['severity_accuracy']:.2f} | {tier0['confidence']:.2f} | {tier0['roof_area_m2']:.1f} |",
        f"| Tier-1 (VHR) | {tier1['gsd_m']} m | {tier1['material_mIoU']:.2f} | {tier1['corrosion_mIoU']:.2f} | {tier1['severity_accuracy']:.2f} | {tier1['confidence']:.2f} | {tier1['roof_area_m2']:.1f} |",
        f"| Tier-3 (Drone) | {tier3['gsd_m']} m | {tier3['material_mIoU']:.2f} | {tier3['corrosion_mIoU']:.2f} | {tier3['severity_accuracy']:.2f} | {tier3['confidence']:.2f} | {tier3['roof_area_m2']:.1f} |",
        "",
        "## Improvements",
        "",
        "### Tier-0 → Tier-1",
        "",
        f"- Material mIoU: +{t0_to_t1['material_mIoU']['absolute']:.4f} ({t0_to_t1['material_mIoU']['relative']:+.1f}%)",
        f"- Corrosion mIoU: +{t0_to_t1['corrosion_mIoU']['absolute']:.4f} ({t0_to_t1['corrosion_mIoU']['relative']:+.1f}%)",
        f"- Severity accuracy: +{t0_to_t1['severity_accuracy']['absolute']:.4f} ({t0_to_t1['severity_accuracy']['relative']:+.1f}%)",
        f"- Confidence: {t0_to_t1['confidence']['absolute']:+.4f}",
        "",
        "### Tier-1 → Tier-3",
        "",
        f"- Material mIoU: +{t1_to_t3['material_mIoU']['absolute']:.4f} ({t1_to_t3['material_mIoU']['relative']:+.1f}%)",
        f"- Corrosion mIoU: +{t1_to_t3['corrosion_mIoU']['absolute']:.4f} ({t1_to_t3['corrosion_mIoU']['relative']:+.1f}%)",
        f"- Severity accuracy: +{t1_to_t3['severity_accuracy']['absolute']:.4f} ({t1_to_t3['severity_accuracy']['relative']:+.1f}%)",
        f"- Confidence: {t1_to_t3['confidence']['absolute']:+.4f}",
        "",
        "## Material Breakdown Comparison",
        "",
        "| Tier | Metal % | Tile % | Other % |",
        "|------|---------|--------|---------|",
        f"| Tier-0 | {tier0['material_breakdown'].get('metal_percent', 0):.1f} | {tier0['material_breakdown'].get('tile_percent', 0):.1f} | {tier0['material_breakdown'].get('other_percent', 0):.1f} |",
        f"| Tier-1 | {tier1['material_breakdown'].get('metal', 0):.1f} | {tier1['material_breakdown'].get('tile', 0):.1f} | {tier1['material_breakdown'].get('other', 0):.1f} |",
        f"| Tier-3 | {tier3['material_breakdown'].get('metal', 0):.1f} | {tier3['material_breakdown'].get('tile', 0):.1f} | {tier3['material_breakdown'].get('other', 0):.1f} |",
        "",
        "## Notes",
        "",
        "- Tier-0 uses 10 m GSD Sentinel-2 (free, ±30% estimate)",
        "- Tier-1 uses 0.5 m GSD Pléiades (€3.80/km², ±15% quote)",
        "- Tier-3 uses 5 cm GSD drone + estimator HITL (binding quote)",
        "- Benchmark values are stubs until real models are trained",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-tier benchmark")
    parser.add_argument("--polygon", default=None, help="Path to GeoJSON polygon file")
    parser.add_argument("--output", default="reports/cross_tier_benchmark.md", help="Output markdown path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    polygon = {"type": "Polygon", "coordinates": [[[100.5, 13.75], [100.51, 13.75], [100.51, 13.76], [100.5, 13.76], [100.5, 13.75]]]}
    if args.polygon:
        with open(args.polygon) as f:
            polygon = json.load(f)

    tier0 = run_tier0(polygon)
    tier1 = run_tier1(polygon)
    tier3 = run_tier3(polygon)

    report = generate_report(polygon, tier0, tier1, tier3)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
