"""Tests for ml/flows/retrain_flow.py"""

import pytest

pytest.importorskip("prefect")

from flows.retrain_flow import (
    fetch_corrections,
    active_learning_prioritize,
    promote_model,
)


def test_fetch_corrections():
    result = fetch_corrections.fn()
    assert "corrections" in result
    assert "count" in result
    assert result["count"] == len(result["corrections"])
    assert "since" in result


def test_active_learning_prioritize():
    corrections = [{"id": "c1"}, {"id": "c2"}]
    result = active_learning_prioritize.fn(corrections, budget=50)
    assert len(result["selected_tile_ids"]) == 50
    assert len(result["acquisition_scores"]) == 50
    assert result["method"] == "BatchBALD-stub"


def test_promote_model_success():
    frozen = {"frozen_material_mIoU": 0.70}
    prod = {"frozen_material_mIoU": 0.65}
    result = promote_model.fn(
        new_model_uri="mlflow:/models/test/1",
        frozen_metrics=frozen,
        production_metrics=prod,
        min_improvement=0.005,
    )
    assert result["promoted"] is True
    assert result["alias"] == "production"


def test_promote_model_failure():
    frozen = {"frozen_material_mIoU": 0.651}
    prod = {"frozen_material_mIoU": 0.65}
    result = promote_model.fn(
        new_model_uri="mlflow:/models/test/1",
        frozen_metrics=frozen,
        production_metrics=prod,
        min_improvement=0.005,
    )
    assert result["promoted"] is False
    assert result["alias"] == "staging"
