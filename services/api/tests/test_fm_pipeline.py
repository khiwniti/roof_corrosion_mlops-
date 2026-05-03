"""Tests for the foundation-model pipeline components.

Mocks external HTTP calls (NIM, OSM) so tests run offline and fast.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.inference.footprint_client import (
    BuildingFootprintClient,
    meters_to_degrees,
    polygon_area_m2,
)
from app.inference.nim_client import NIMVisionClient, NIMError


# ═══════════════════════════════════════════════════════════════
# Footprint client unit tests (pure geometry, no network)
# ═══════════════════════════════════════════════════════════════

def test_meters_to_degrees_at_equator():
    lat_deg, lng_deg = meters_to_degrees(1000.0, lat=0.0)
    # ~1/111 deg per km at equator
    assert 0.008 < lat_deg < 0.010
    assert 0.008 < lng_deg < 0.010


def test_meters_to_degrees_at_high_latitude():
    lat_deg, lng_deg = meters_to_degrees(1000.0, lat=60.0)
    # lng_deg should be ~2x lat_deg at lat=60 since cos(60)=0.5
    assert abs(lng_deg - 2 * lat_deg) < 1e-3


def test_polygon_area_unit_square():
    # Tiny square ~1m x 1m at NYC latitude
    m_per_deg_lat = 111_320.0
    d = 1.0 / m_per_deg_lat
    polygon = [[40.0, -74.0], [40.0 + d, -74.0], [40.0 + d, -74.0 + d], [40.0, -74.0 + d]]
    area = polygon_area_m2(polygon)
    # Should be approximately 1 m² (within 10% due to projection)
    assert 0.5 < area < 1.5


def test_polygon_area_degenerate():
    assert polygon_area_m2([]) == 0.0
    assert polygon_area_m2([[40.0, -74.0]]) == 0.0
    assert polygon_area_m2([[40.0, -74.0], [40.1, -74.0]]) == 0.0


def test_bbox_fallback_has_correct_shape():
    client = BuildingFootprintClient()
    fp = client._make_bbox_footprint(lat=40.0, lng=-74.0, radius_m=30.0)
    assert fp["source"] == "bbox"
    assert len(fp["polygon_ll"]) == 5  # closed ring
    assert fp["area_m2"] > 0


# ═══════════════════════════════════════════════════════════════
# NIM client unit tests (mocked network)
# ═══════════════════════════════════════════════════════════════

def test_nim_requires_api_key():
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NIMVisionClient(api_key="", base_url="https://integrate.api.nvidia.com/v1")


def test_nim_allows_self_hosted_without_key():
    # Self-hosted NIM doesn't need a key
    client = NIMVisionClient(api_key="", base_url="http://localhost:8000/v1")
    assert client.base_url == "http://localhost:8000/v1"


def test_nim_parse_json_valid():
    raw = '{"corrosion_percent": 15.0, "severity": "light", "confidence": 0.8}'
    result = NIMVisionClient._parse_json(raw)
    assert result["corrosion_percent"] == 15.0
    assert result["severity"] == "light"


def test_nim_parse_json_with_markdown_fences():
    raw = '```json\n{"corrosion_percent": 20.0}\n```'
    result = NIMVisionClient._parse_json(raw)
    assert result["corrosion_percent"] == 20.0


def test_nim_parse_json_with_leading_prose():
    raw = 'Here is my assessment:\n{"corrosion_percent": 5.0, "severity": "light"}\nThank you.'
    result = NIMVisionClient._parse_json(raw)
    assert result["severity"] == "light"


def test_nim_parse_json_invalid_raises():
    with pytest.raises(NIMError):
        NIMVisionClient._parse_json("this is not json at all")


def test_nim_normalize_clamps_ranges():
    raw = {"corrosion_percent": 150.0, "confidence": 2.5, "severity": "none"}
    result = NIMVisionClient._normalize_result(raw)
    assert result["corrosion_percent"] == 100.0
    assert result["confidence"] == 1.0


def test_nim_normalize_derives_severity_from_percent():
    raw = {"corrosion_percent": 30.0, "severity": "invalid_value"}
    result = NIMVisionClient._normalize_result(raw)
    assert result["severity"] == "moderate"  # 30% → moderate (25-50%)


def test_nim_normalize_fills_defaults():
    result = NIMVisionClient._normalize_result({})
    assert result["assessable"] is True
    assert result["corrosion_percent"] == 0.0
    assert result["severity"] == "none"
    assert result["visible_issues"] == []


def test_nim_image_to_data_uri():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    uri = NIMVisionClient._to_data_uri(img)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 100


# ═══════════════════════════════════════════════════════════════
# NIM client integration tests (mocked HTTP)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_nim_assess_corrosion_mocked():
    """Full assess_corrosion call with mocked HTTP response."""
    client = NIMVisionClient(api_key="fake-key")

    mock_response_content = json.dumps({
        "assessable": True,
        "roof_material": "metal_corrugated",
        "corrosion_percent": 22.5,
        "severity": "light",
        "confidence": 0.85,
        "description": "Scattered rust patches on the northern slope.",
        "rationale": "Orange discoloration visible across ~22% of metal panels.",
        "visible_issues": ["rust_patches", "coating_failure"]
    })

    mock_api_response = {
        "choices": [{"message": {"content": mock_response_content}}]
    }

    tile = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    with patch.object(client, "_chat_completion", new=AsyncMock(return_value=mock_response_content)):
        result = await client.assess_corrosion(tile, gsd=0.3, address="123 Main St")

    assert result["corrosion_percent"] == 22.5
    assert result["severity"] == "light"
    assert result["roof_material"] == "metal_corrugated"
    assert result["confidence"] == 0.85
    assert "rust_patches" in result["visible_issues"]


@pytest.mark.asyncio
async def test_nim_assess_corrosion_handles_bad_json():
    client = NIMVisionClient(api_key="fake-key")
    tile = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    with patch.object(client, "_chat_completion", new=AsyncMock(return_value="not valid json")):
        with pytest.raises(NIMError):
            await client.assess_corrosion(tile, gsd=0.3)
