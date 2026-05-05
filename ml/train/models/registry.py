"""Model registry for loading and saving trained checkpoints.

Bridges Phase 2 training (MLflow logging) and Phase 3 inference
(RunPod serverless loading). Supports:
- MLflow Model Registry (staging / production aliases)
- Local filesystem checkpoints
- S3 / Cloudflare R2 URI (s3://bucket/path)

Architecture: ADR-011, ADR-009
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("train.models.registry")


def _parse_uri(uri: str) -> tuple[str, str]:
    """Parse a model URI into (scheme, path).

    Examples:
        mlflow:/models/roof-corrosion/Production → ("mlflow", "models/roof-corrosion/Production")
        s3://bucket/checkpoints/model.pt → ("s3", "bucket/checkpoints/model.pt")
        /local/path/checkpoint.pt → ("file", "/local/path/checkpoint.pt")
    """
    if uri.startswith("mlflow:"):
        return "mlflow", uri[len("mlflow:") :].lstrip("/")
    if uri.startswith("s3://") or uri.startswith("r2://"):
        scheme = uri.split("://", 1)[0]
        return scheme, uri.split("://", 1)[1]
    if "://" in uri:
        scheme, path = uri.split("://", 1)
        return scheme, path
    return "file", uri


def load_checkpoint(
    uri: str,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a PyTorch checkpoint from URI.

    Returns the raw checkpoint dict (usually contains 'state_dict',
    'hyper_parameters', 'epoch', etc.).
    """
    scheme, path = _parse_uri(uri)
    logger.info("Loading checkpoint from %s (scheme=%s)", uri, scheme)

    if scheme == "file":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if scheme == "s3" or scheme == "r2":
        # Lazy import boto3 only when needed
        import boto3

        bucket, key = path.split("/", 1)
        s3 = boto3.client("s3")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            s3.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name
        try:
            checkpoint = torch.load(tmp_path, map_location=map_location, weights_only=False)
        finally:
            os.unlink(tmp_path)
        return checkpoint

    if scheme == "mlflow":
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError as e:
            raise RuntimeError("mlflow not installed; cannot load from MLflow URI") from e

        # Parse alias or version, e.g. "models/roof-corrosion/Production"
        parts = path.split("/")
        model_name = "/".join(parts[:-1]) if len(parts) > 1 else path
        alias_or_version = parts[-1] if len(parts) > 1 else "latest"

        # MLflow 3.x alias API
        client = mlflow.tracking.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(model_name, alias_or_version)
        except Exception:
            # Fallback: treat as version number
            mv = client.get_model_version(model_name, alias_or_version)

        artifact_uri = mv.source
        logger.info("Resolved MLflow artifact URI: %s", artifact_uri)
        # Recurse to load from the resolved URI (usually file:// or s3://)
        return load_checkpoint(artifact_uri, map_location=map_location)

    raise ValueError(f"Unsupported scheme: {scheme}")


def load_model(
    uri: str,
    model_class: type[nn.Module] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a full model (architecture + weights) from URI.

    If model_class is None, attempts to reconstruct from checkpoint
    hyper_parameters (requires checkpoint to contain 'hyper_parameters'
    and 'state_dict').
    """
    checkpoint = load_checkpoint(uri, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if model_class is not None:
        kwargs = model_kwargs or {}
        model = model_class(**kwargs)
        model.load_state_dict(state_dict, strict=strict)
        return model

    # Try to reconstruct from hyper_parameters
    hparams = checkpoint.get("hyper_parameters", {})
    if not hparams:
        raise ValueError(
            "model_class is None and checkpoint has no 'hyper_parameters'. "
            "Please provide model_class explicitly."
        )

    # Reconstruct ClayMultiTaskModel from hparams
    from ml.train.models.clay_multitask import ClayMultiTaskModel

    model = ClayMultiTaskModel(**hparams)
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_checkpoint(
    model: nn.Module,
    uri: str,
    hyper_parameters: dict[str, Any] | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Save a PyTorch checkpoint to URI.

    Returns the resolved URI where the checkpoint was saved.
    """
    scheme, path = _parse_uri(uri)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": hyper_parameters or {},
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics

    if scheme == "file":
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out_path)
        logger.info("Saved checkpoint to %s", out_path)
        return str(out_path)

    if scheme == "s3" or scheme == "r2":
        import boto3
        import tempfile

        bucket, key = path.split("/", 1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(checkpoint, tmp)
            tmp_path = tmp.name
        try:
            s3 = boto3.client("s3")
            s3.upload_file(tmp_path, bucket, key)
            logger.info("Saved checkpoint to s3://%s/%s", bucket, key)
        finally:
            os.unlink(tmp_path)
        return uri

    if scheme == "mlflow":
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError as e:
            raise RuntimeError("mlflow not installed; cannot save to MLflow URI") from e

        parts = path.split("/")
        model_name = "/".join(parts) if len(parts) > 1 else path

        with mlflow.start_run():
            mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=model_name)
            if metrics:
                mlflow.log_metrics(metrics)
        logger.info("Saved model to MLflow registry: %s", model_name)
        return f"mlflow:/{model_name}"

    raise ValueError(f"Unsupported scheme: {scheme}")


def list_local_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List all .pt / .ckpt files in a directory, sorted by mtime."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    files += sorted(d.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files
