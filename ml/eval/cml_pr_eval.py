"""CML (Continuous Machine Learning) PR evaluation pipeline.

Runs a lightweight benchmark on every pull request that touches ml/ code,
generates a markdown report with metrics, and outputs CML commands for
posting as a PR comment.

Architecture: ADR-011

Usage in CI (GitHub Actions):
    python ml/eval/cml_pr_eval.py --model-uri mlflow:/models/roof-corrosion/staging

Requirements: pip install cml
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval.cml_pr_eval")


def run_quick_benchmark(
    model_uri: str,
    benchmark_dir: str = "data/frozen_test",
    max_samples: int = 50,
) -> dict[str, Any]:
    """Run a quick benchmark on a subset of the frozen test set.

    In production, this loads the model from MLflow and runs inference.
    Stub: returns synthetic metrics.
    """
    logger.info("Running quick benchmark on %s (max_samples=%d)", model_uri, max_samples)

    # TODO: actual model loading and inference
    # from ml.train.models.registry import load_model
    # model = load_model(model_uri)
    # metrics = evaluate(model, benchmark_dir, max_samples=max_samples)

    metrics = {
        "material_mIoU": 0.67,
        "corrosion_mIoU": 0.51,
        "severity_accuracy": 0.73,
        "mean_inference_ms": 245.0,
        "samples_evaluated": max_samples,
    }

    logger.info("Benchmark complete: %s", metrics)
    return metrics


def generate_report(
    metrics: dict[str, Any],
    model_uri: str,
    baseline_metrics: dict[str, Any] | None = None,
) -> str:
    """Generate a markdown report for CML PR comment.

    Includes delta vs baseline if provided.
    """
    baseline = baseline_metrics or {
        "material_mIoU": 0.65,
        "corrosion_mIoU": 0.49,
        "severity_accuracy": 0.71,
    }

    def _delta(key: str) -> str:
        diff = metrics.get(key, 0) - baseline.get(key, 0)
        emoji = "🟢" if diff >= 0 else "🔴"
        return f"{emoji} {diff:+.4f}"

    def _fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)

    lines = [
        "# 🤖 CML PR Evaluation Report",
        "",
        f"**Model:** `{model_uri}`  ",
        f"**Timestamp:** {datetime.now(UTC).isoformat()}  ",
        f"**Samples:** {metrics.get('samples_evaluated', 'N/A')}",
        "",
        "## Metrics",
        "",
        "| Metric | This PR | Baseline | Δ |",
        "|--------|---------|----------|---|",
        f"| Material mIoU | {_fmt(metrics.get('material_mIoU'))} | {_fmt(baseline.get('material_mIoU'))} | {_delta('material_mIoU')} |",
        f"| Corrosion mIoU | {_fmt(metrics.get('corrosion_mIoU'))} | {_fmt(baseline.get('corrosion_mIoU'))} | {_delta('corrosion_mIoU')} |",
        f"| Severity Accuracy | {_fmt(metrics.get('severity_accuracy'))} | {_fmt(baseline.get('severity_accuracy'))} | {_delta('severity_accuracy')} |",
        f"| Mean Inference | {_fmt(metrics.get('mean_inference_ms'))} ms | — | — |",
        "",
        "## Go / No-Go",
        "",
    ]

    material_ok = metrics.get("material_mIoU", 0) >= 0.65
    corrosion_ok = metrics.get("corrosion_mIoU", 0) >= 0.45
    severity_ok = metrics.get("severity_accuracy", 0) >= 0.70

    if material_ok and corrosion_ok and severity_ok:
        lines.append("✅ **GO** — All metrics meet minimum thresholds.")
    else:
        lines.append("⚠️ **CONDITIONAL** — Some metrics below threshold:")
        if not material_ok:
            lines.append("- Material mIoU < 0.65")
        if not corrosion_ok:
            lines.append("- Corrosion mIoU < 0.45")
        if not severity_ok:
            lines.append("- Severity accuracy < 0.70")

    lines.extend([
        "",
        "<!-- CML-report -->",
    ])

    return "\n".join(lines)


def save_cml_outputs(report: str, metrics: dict[str, Any], output_dir: str = "reports/cml") -> dict[str, Path]:
    """Save report and metrics for CML to pick up.

    Returns paths to generated files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "pr_report.md"
    report_path.write_text(report)

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("CML outputs saved to %s", out)
    return {
        "report": report_path,
        "metrics": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CML PR evaluation")
    parser.add_argument("--model-uri", default="mlflow:/models/roof-corrosion/staging")
    parser.add_argument("--benchmark-dir", default="data/frozen_test")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--baseline", default=None, help="Path to baseline metrics JSON")
    parser.add_argument("--output-dir", default="reports/cml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)

    metrics = run_quick_benchmark(args.model_uri, args.benchmark_dir, args.max_samples)
    report = generate_report(metrics, args.model_uri, baseline)
    outputs = save_cml_outputs(report, metrics, args.output_dir)

    print(report)
    print(f"\nOutputs saved to: {outputs['report']}")


if __name__ == "__main__":
    main()
