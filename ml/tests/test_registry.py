"""Tests for ml/train/models/registry.py"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from train.models.registry import (
    _parse_uri,
    load_checkpoint,
    save_checkpoint,
    list_local_checkpoints,
)


def test_parse_uri():
    assert _parse_uri("mlflow:/models/roof-corrosion/Production") == ("mlflow", "models/roof-corrosion/Production")
    assert _parse_uri("s3://bucket/checkpoints/model.pt") == ("s3", "bucket/checkpoints/model.pt")
    assert _parse_uri("/local/path/model.pt") == ("file", "/local/path/model.pt")
    assert _parse_uri("file:///local/path/model.pt") == ("file", "/local/path/model.pt")


def test_save_and_load_checkpoint():
    model = nn.Linear(10, 2)
    hparams = {"in_features": 10, "out_features": 2}

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = f"{tmpdir}/test_checkpoint.pt"
        save_checkpoint(model, uri, hyper_parameters=hparams, epoch=5, metrics={"val_mIoU": 0.65})

        ckpt = load_checkpoint(uri)
        assert "state_dict" in ckpt
        assert ckpt["hyper_parameters"] == hparams
        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["val_mIoU"] == 0.65


def test_list_local_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(f"{tmpdir}/model_v1.pt").touch()
        Path(f"{tmpdir}/model_v2.pt").touch()
        ckpts = list_local_checkpoints(tmpdir)
        assert len(ckpts) == 2
        assert all(p.suffix == ".pt" for p in ckpts)
