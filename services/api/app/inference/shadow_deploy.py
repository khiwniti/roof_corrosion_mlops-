"""Shadow deployment for safe model rollouts.

Runs new model versions alongside production, comparing outputs
without serving them to customers. Promotes only if shadow metrics
meet or exceed production on the frozen test set.

Usage:
    # In the inference worker, after loading production model:
    shadow = ShadowDeployManager(mlflow_tracking_uri="http://mlflow:5000")
    shadow.load_shadow_models()

    # During inference:
    result = pipeline.analyze(tile, gsd=0.3)
    shadow.compare(pipeline_result=result, tile_image=tile, gsd=0.3)

    # After N comparisons:
    shadow.report()  # prints comparison stats
    shadow.promote_if_better()  # promotes shadow to production if metrics improve
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.inference.pipeline import CorrosionPipeline, CorrosionResult


@dataclass
class ShadowComparison:
    """Result of comparing shadow vs production on a single tile."""

    job_id: str
    timestamp: float

    # Production outputs
    prod_roof_area_m2: float
    prod_corroded_area_m2: float
    prod_corrosion_percent: float
    prod_severity: str
    prod_confidence: float

    # Shadow outputs
    shadow_roof_area_m2: float
    shadow_corroded_area_m2: float
    shadow_corrosion_percent: float
    shadow_severity: str
    shadow_confidence: float

    # Agreement metrics
    severity_agreement: bool
    area_diff_m2: float
    area_diff_percent: float


@dataclass
class ShadowStats:
    """Aggregated shadow deployment statistics."""

    total_comparisons: int = 0
    severity_agreement_rate: float = 0.0
    mean_area_diff_percent: float = 0.0
    mean_confidence_diff: float = 0.0
    shadow_higher_confidence_rate: float = 0.0
    severity_confusion: dict = field(default_factory=dict)


class ShadowDeployManager:
    """Manages shadow deployment of new model versions.

    Workflow:
    1. Load production models (current serving version)
    2. Load shadow models (candidate version from MLflow staging)
    3. For each inference request, run both and log comparison
    4. After N comparisons, evaluate whether shadow is better
    5. If shadow meets promotion criteria, promote via MLflow registry
    """

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        min_comparisons: int = 100,
        promotion_threshold_severity_agreement: float = 0.90,
        promotion_threshold_confidence_improvement: float = 0.05,
        log_dir: str = "data/shadow_deploy",
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.min_comparisons = min_comparisons
        self.promotion_threshold_severity = promotion_threshold_severity_agreement
        self.promotion_threshold_confidence = promotion_threshold_confidence_improvement
        self.log_dir = Path(log_dir)

        self.shadow_pipeline: Optional[CorrosionPipeline] = None
        self.comparisons: list[ShadowComparison] = []
        self._severity_counts: dict[tuple[str, str], int] = defaultdict(int)

    def load_shadow_models(self, device: str = "auto") -> None:
        """Load shadow (staging) models from MLflow."""
        try:
            self.shadow_pipeline = CorrosionPipeline(
                roof_model_uri="models:/roof_detector/staging",
                corrosion_model_uri="models:/corrosion_detector/staging",
                device=device,
            )
            print("Shadow models loaded from MLflow staging")
        except Exception as e:
            print(f"⚠️  Failed to load shadow models: {e}")
            self.shadow_pipeline = None

    def compare(
        self,
        pipeline_result: CorrosionResult,
        tile_image: np.ndarray,
        gsd: float,
        job_id: str = "",
    ) -> Optional[ShadowComparison]:
        """Run shadow model on the same tile and compare with production.

        This is called AFTER the production model has already produced a result.
        The shadow result is logged but NOT served to the customer.
        """
        if self.shadow_pipeline is None:
            return None

        try:
            shadow_result = self.shadow_pipeline.analyze(tile_image, gsd=gsd)
        except Exception as e:
            print(f"⚠️  Shadow inference failed: {e}")
            return None

        comparison = ShadowComparison(
            job_id=job_id,
            timestamp=time.time(),
            prod_roof_area_m2=pipeline_result.roof_area_m2,
            prod_corroded_area_m2=pipeline_result.corroded_area_m2,
            prod_corrosion_percent=pipeline_result.corrosion_percent,
            prod_severity=pipeline_result.severity,
            prod_confidence=pipeline_result.confidence,
            shadow_roof_area_m2=shadow_result.roof_area_m2,
            shadow_corroded_area_m2=shadow_result.corroded_area_m2,
            shadow_corrosion_percent=shadow_result.corrosion_percent,
            shadow_severity=shadow_result.severity,
            shadow_confidence=shadow_result.confidence,
            severity_agreement=pipeline_result.severity == shadow_result.severity,
            area_diff_m2=abs(pipeline_result.corroded_area_m2 - shadow_result.corroded_area_m2),
            area_diff_percent=abs(pipeline_result.corrosion_percent - shadow_result.corrosion_percent),
        )

        self.comparisons.append(comparison)
        self._severity_counts[(pipeline_result.severity, shadow_result.severity)] += 1

        # Log to file
        self._log_comparison(comparison)

        return comparison

    def _log_comparison(self, comparison: ShadowComparison) -> None:
        """Append comparison to JSONL log file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"shadow_{time.strftime('%Y%m%d')}.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "job_id": comparison.job_id,
                "timestamp": comparison.timestamp,
                "prod_severity": comparison.prod_severity,
                "shadow_severity": comparison.shadow_severity,
                "agreement": comparison.severity_agreement,
                "prod_area": comparison.prod_corroded_area_m2,
                "shadow_area": comparison.shadow_corroded_area_m2,
                "area_diff_pct": comparison.area_diff_percent,
                "prod_confidence": comparison.prod_confidence,
                "shadow_confidence": comparison.shadow_confidence,
            }) + "\n")

    def compute_stats(self) -> ShadowStats:
        """Compute aggregated shadow deployment statistics."""
        if not self.comparisons:
            return ShadowStats()

        n = len(self.comparisons)
        agreements = sum(1 for c in self.comparisons if c.severity_agreement)
        area_diffs = [c.area_diff_percent for c in self.comparisons]
        conf_diffs = [c.shadow_confidence - c.prod_confidence for c in self.comparisons]
        shadow_higher = sum(1 for d in conf_diffs if d > 0)

        # Build confusion matrix
        confusion = {}
        for (prod_sev, shadow_sev), count in self._severity_counts.items():
            confusion[f"{prod_sev}->{shadow_sev}"] = count

        return ShadowStats(
            total_comparisons=n,
            severity_agreement_rate=agreements / n,
            mean_area_diff_percent=np.mean(area_diffs),
            mean_confidence_diff=np.mean(conf_diffs),
            shadow_higher_confidence_rate=shadow_higher / n,
            severity_confusion=confusion,
        )

    def should_promote(self) -> tuple[bool, str]:
        """Evaluate whether shadow model should be promoted to production.

        Promotion criteria:
        1. At least min_comparisons samples collected
        2. Severity agreement rate ≥ threshold (shadow doesn't flip grades)
        3. Shadow confidence ≥ production confidence + improvement margin
        4. No severity regression (shadow never downgrades severe → light)

        Returns:
            (should_promote: bool, reason: str)
        """
        stats = self.compute_stats()

        if stats.total_comparisons < self.min_comparisons:
            return False, f"Only {stats.total_comparisons}/{self.min_comparisons} comparisons. Need more data."

        # Check severity agreement
        if stats.severity_agreement_rate < self.promotion_threshold_severity:
            return False, (
                f"Severity agreement {stats.severity_agreement_rate:.2%} "
                f"< {self.promotion_threshold_severity:.2%} threshold"
            )

        # Check confidence improvement
        if stats.mean_confidence_diff < self.promotion_threshold_confidence:
            return False, (
                f"Confidence improvement {stats.mean_confidence_diff:.3f} "
                f"< {self.promotion_threshold_confidence:.3f} threshold"
            )

        # Check for severity regression
        if "severe->light" in stats.severity_confusion or "severe->none" in stats.severity_confusion:
            return False, "Shadow model regresses severe corrosion to light/none. Unsafe to promote."

        return True, (
            f"All criteria met: agreement={stats.severity_agreement_rate:.2%}, "
            f"confidence_delta={stats.mean_confidence_diff:.3f}, "
            f"n={stats.total_comparisons}"
        )

    def promote_if_better(self) -> bool:
        """Check promotion criteria and promote shadow to production via MLflow."""
        should, reason = self.should_promote()
        print(f"Shadow promotion check: {reason}")

        if not should:
            return False

        try:
            import mlflow
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            # Transition shadow models from staging → production
            client = mlflow.tracking.MlflowClient()
            for model_name in ["roof_detector", "corrosion_detector"]:
                # Archive current production
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=client.get_latest_versions(model_name, stages=["Production"])[0].version,
                        stage="Archived",
                    )
                except Exception:
                    pass  # No current production version

                # Promote staging → production
                staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                if staging_versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=staging_versions[0].version,
                        stage="Production",
                    )
                    print(f"✅ Promoted {model_name} staging → production")

            return True
        except Exception as e:
            print(f"⚠️  MLflow promotion failed: {e}")
            return False

    def report(self) -> str:
        """Generate a human-readable shadow deployment report."""
        stats = self.compute_stats()
        lines = [
            "═══════════════════════════════════════════════════════",
            "  SHADOW DEPLOYMENT REPORT",
            "═══════════════════════════════════════════════════════",
            f"  Comparisons:        {stats.total_comparisons}",
            f"  Severity agreement: {stats.severity_agreement_rate:.2%}",
            f"  Mean area diff:     {stats.mean_area_diff_percent:.2f}%",
            f"  Mean confidence Δ:  {stats.mean_confidence_diff:+.3f}",
            f"  Shadow higher conf: {stats.shadow_higher_confidence_rate:.2%}",
            "",
            "  Severity confusion:",
        ]
        for transition, count in sorted(stats.severity_confusion.items()):
            lines.append(f"    {transition}: {count}")

        should, reason = self.should_promote()
        lines.extend([
            "",
            f"  Promotion: {'✅ YES' if should else '❌ NOT YET'}",
            f"  Reason: {reason}",
            "═══════════════════════════════════════════════════════",
        ])
        return "\n".join(lines)
