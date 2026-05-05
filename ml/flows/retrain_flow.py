"""Prefect orchestration flow for HITL flywheel retraining (Phase 4/5).

Collects estimator corrections, runs active-learning prioritization,
triggers Clay + Mask2Former multi-task retraining, evaluates on frozen
set, and promotes via MLflow alias swap.

Usage:
    python ml/flows/retrain_flow.py
    # or via Prefect UI:
    prefect deploy -n hitl-retrain-flow

Architecture: ADR-011, ADR-013
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from prefect import flow, task, get_run_logger


@task(retries=1, retry_delay_seconds=30, timeout_seconds=600)
def fetch_corrections(since: datetime | None = None) -> dict[str, Any]:
    """Fetch estimator corrections from Label Studio / Supabase.

    Returns dict with:
        - corrections: list of correction records
        - count: number of new corrections
        - since: timestamp of last check
    """
    logger = get_run_logger()
    since = since or datetime.now(UTC) - timedelta(days=7)
    logger.info("Fetching corrections since %s", since.isoformat())

    # TODO: integrate with Label Studio API or Supabase corrections table
    # Stub: return synthetic corrections for now
    corrections = [
        {
            "id": f"corr_{i}",
            "job_id": f"job_{i}",
            "field": "material",
            "old_value": "metal",
            "new_value": "tile",
            "weight": 3.0,
            "created_at": datetime.now(UTC).isoformat(),
        }
        for i in range(5)
    ]

    logger.info("Fetched %d corrections", len(corrections))
    return {"corrections": corrections, "count": len(corrections), "since": since.isoformat()}


@task(timeout_seconds=1800)
def active_learning_prioritize(
    corrections: list[dict],
    unlabeled_pool_size: int = 1000,
    budget: int = 200,
) -> dict[str, Any]:
    """Run BatchBALD / MC-Dropout to prioritize most informative tiles.

    Parameters
    ----------
    corrections : list of correction dicts (used as labeled pool)
    unlabeled_pool_size : total unlabeled tiles available
    budget : number of tiles to select for labeling

    Returns
    -------
    dict with selected_tile_ids, acquisition_scores, method
    """
    logger = get_run_logger()
    logger.info("Running active learning prioritization (budget=%d)", budget)

    # TODO: integrate with baal library for real BatchBALD
    # Stub: random selection weighted by correction count
    import random

    random.seed(42)
    selected = [f"tile_{random.randint(0, unlabeled_pool_size - 1)}" for _ in range(budget)]
    scores = [random.random() for _ in range(budget)]

    logger.info("Selected %d tiles for labeling", len(selected))
    return {
        "selected_tile_ids": selected,
        "acquisition_scores": scores,
        "method": "BatchBALD-stub",
        "budget": budget,
    }


@task(retries=1, retry_delay_seconds=60, timeout_seconds=14400)
def retrain_model(
    training_data_path: str,
    config_path: str = "ml/train/configs/clay_multitask.yaml",
    model_uri: str | None = None,
) -> dict[str, Any]:
    """Retrain Clay + Mask2Former multi-task model.

    If model_uri is provided, resume from that checkpoint (warm start).
    Otherwise train from scratch.

    Returns dict with mlflow_run_id, model_uri, val_metrics.
    """
    logger = get_run_logger()
    logger.info("Starting retraining with config: %s", config_path)
    if model_uri:
        logger.info("Warm-starting from %s", model_uri)

    # TODO: actual training integration
    # python ml/train/train_multitask.py --config config_path --resume-from model_uri
    import subprocess

    result = subprocess.run(
        ["python", "ml/train/train_multitask.py", "--config", config_path],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )

    if result.returncode != 0:
        logger.error("Training failed:\n%s", result.stderr)
        raise RuntimeError(f"Training failed: {result.stderr[:500]}")

    # Stub metrics
    val_metrics = {
        "val_material_mIoU": 0.68,
        "val_corrosion_mIoU": 0.52,
        "val_severity_accuracy": 0.74,
    }

    mlflow_run_id = "stub-run-id"
    new_model_uri = f"mlflow:/models/roof-corrosion/{mlflow_run_id}"

    logger.info("Training complete. val_material_mIoU=%.3f", val_metrics["val_material_mIoU"])
    return {
        "mlflow_run_id": mlflow_run_id,
        "model_uri": new_model_uri,
        "val_metrics": val_metrics,
    }


@task(timeout_seconds=1800)
def evaluate_on_frozen(
    model_uri: str,
    frozen_dir: str = "data/frozen_test",
) -> dict[str, Any]:
    """Evaluate retrained model on frozen real-image test set.

    Returns frozen_metrics dict.
    """
    logger = get_run_logger()
    logger.info("Evaluating %s on frozen set %s", model_uri, frozen_dir)

    # TODO: actual frozen evaluation
    frozen_metrics = {
        "frozen_material_mIoU": 0.66,
        "frozen_corrosion_mIoU": 0.50,
        "frozen_severity_accuracy": 0.72,
    }

    logger.info("Frozen eval complete: %s", frozen_metrics)
    return frozen_metrics


@task(timeout_seconds=300)
def promote_model(
    new_model_uri: str,
    frozen_metrics: dict[str, float],
    production_metrics: dict[str, float] | None = None,
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    """Promote model if frozen metrics improve over production.

    Uses MLflow alias swap: staging → production.

    Parameters
    ----------
    new_model_uri : URI of newly trained model
    frozen_metrics : dict of frozen test metrics
    production_metrics : current production model metrics (optional)
    min_improvement : minimum improvement (pp) required for promotion

    Returns
    -------
    dict with promoted, reason, alias
    """
    logger = get_run_logger()

    # Default production metrics if not provided
    prod = production_metrics or {
        "frozen_material_mIoU": 0.65,
        "frozen_corrosion_mIoU": 0.49,
        "frozen_severity_accuracy": 0.71,
    }

    improvement = (
        frozen_metrics.get("frozen_material_mIoU", 0)
        - prod.get("frozen_material_mIoU", 0)
    )

    if improvement >= min_improvement:
        logger.info("Promoting model (improvement=%.3f pp)", improvement * 100)
        # TODO: actual MLflow alias swap
        # mlflow_client.set_registered_model_alias("roof-corrosion", "production", version)
        return {
            "promoted": True,
            "reason": f"mIoU improved by {improvement:.4f}",
            "alias": "production",
            "model_uri": new_model_uri,
        }

    logger.warning(
        "Model NOT promoted (improvement=%.3f pp < %.3f)",
        improvement * 100,
        min_improvement * 100,
    )
    return {
        "promoted": False,
        "reason": f"mIoU improvement {improvement:.4f} below threshold {min_improvement}",
        "alias": "staging",
        "model_uri": new_model_uri,
    }


@task(timeout_seconds=300)
def update_label_studio_queue(selected_tile_ids: list[str]) -> dict[str, Any]:
    """Push newly selected tiles to Label Studio for human labeling.

    Returns status dict.
    """
    logger = get_run_logger()
    logger.info("Updating Label Studio queue with %d tiles", len(selected_tile_ids))

    # TODO: Label Studio API integration
    return {
        "status": "queued",
        "queued_count": len(selected_tile_ids),
        "label_studio_project_id": "stub-project",
    }


@flow(name="hitl-retrain-flow", version="1.0.0")
def hitl_retrain_flow(
    force: bool = False,
    correction_threshold: int = 200,
    active_learning_budget: int = 200,
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    """Orchestrate the HITL flywheel retraining loop.

    Steps:
    1. Fetch corrections since last run
    2. If corrections >= threshold OR force=True:
       a. Run active learning prioritization
       b. Update Label Studio queue
       c. Retrain model (warm start from production)
       d. Evaluate on frozen test set
       e. Promote if improvement >= min_improvement
    3. Report status

    Parameters
    ----------
    force : bypass correction threshold and trigger retrain
    correction_threshold : minimum corrections to trigger retrain
    active_learning_budget : tiles to prioritize for labeling
    min_improvement : minimum mIoU improvement (pp) for promotion
    """
    logger = get_run_logger()
    logger.info("Starting HITL retrain flow (force=%s)", force)

    # Step 1: Fetch corrections
    correction_result = fetch_corrections()
    correction_count = correction_result["count"]

    should_retrain = force or correction_count >= correction_threshold
    if not should_retrain:
        logger.info(
            "Skipping retrain: %d corrections < threshold %d",
            correction_count,
            correction_threshold,
        )
        return {
            "status": "skipped",
            "reason": f"Only {correction_count} corrections (< {correction_threshold})",
            "correction_count": correction_count,
        }

    # Step 2: Active learning prioritization
    al_result = active_learning_prioritize(
        corrections=correction_result["corrections"],
        budget=active_learning_budget,
    )

    # Step 3: Update Label Studio queue
    queue_result = update_label_studio_queue(al_result["selected_tile_ids"])

    # Step 4: Retrain
    train_result = retrain_model(
        training_data_path="data/hitl/training_set",
        model_uri="mlflow:/models/roof-corrosion/production",
    )

    # Step 5: Evaluate on frozen set
    frozen_metrics = evaluate_on_frozen(train_result["model_uri"])

    # Step 6: Promote if improved
    promotion = promote_model(
        new_model_uri=train_result["model_uri"],
        frozen_metrics=frozen_metrics,
        min_improvement=min_improvement,
    )

    report = {
        "status": "complete",
        "correction_count": correction_count,
        "active_learning": al_result,
        "label_studio_queue": queue_result,
        "training": train_result,
        "frozen_eval": frozen_metrics,
        "promotion": promotion,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    logger.info("HITL retrain flow complete. Promoted=%s", promotion["promoted"])
    return report


if __name__ == "__main__":
    hitl_retrain_flow()
