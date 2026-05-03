"""Tests for tile fetcher (Esri/Mapbox/Maxar) and sync /quote/sync endpoint."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.inference.tile_fetch import TileFetcher, TileRequest, ZOOM_TO_GSD_M
from app.main import app


# ═══════════════════════════════════════════════════════════════
# Tile fetcher unit tests
# ═══════════════════════════════════════════════════════════════

def test_auto_select_defaults_to_esri(monkeypatch):
    """With no API keys, Esri (free) is the auto-selected source."""
    for k in ("NEARMAP_API_KEY", "MAXAR_API_KEY", "MAPBOX_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert TileFetcher.auto_select_source() == "esri"


def test_auto_select_prefers_paid_keys(monkeypatch):
    """Priority: nearmap > maxar > mapbox > esri."""
    for k in ("NEARMAP_API_KEY", "MAXAR_API_KEY", "MAPBOX_TOKEN"):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("MAPBOX_TOKEN", "pk.fake")
    assert TileFetcher.auto_select_source() == "mapbox"

    monkeypatch.setenv("MAXAR_API_KEY", "fake")
    assert TileFetcher.auto_select_source() == "maxar"

    monkeypatch.setenv("NEARMAP_API_KEY", "fake")
    assert TileFetcher.auto_select_source() == "nearmap"


def test_gsd_for_zoom():
    """At zoom 19, GSD ~ 0.30m at the equator."""
    assert TileFetcher.gsd_for("esri", 19, lat=0.0) == pytest.approx(0.30, abs=0.01)
    # At Bangkok (~13.7°N), GSD is ~3% smaller
    bangkok_gsd = TileFetcher.gsd_for("esri", 19, lat=13.7)
    assert bangkok_gsd < 0.30
    assert bangkok_gsd > 0.27


def test_gsd_for_unknown_zoom_returns_default():
    """Unknown zooms fall back to 0.30m (zoom 19 default)."""
    assert TileFetcher.gsd_for("esri", 99, lat=0.0) == 0.30


def test_tile_coords_at_origin():
    """Tile coords at (0,0) at zoom 0 should be (0, 0)."""
    fetcher = TileFetcher()
    x, y = fetcher._tile_coords(0.0, 0.0, 0)
    assert (x, y) == (0, 0)


def test_tile_coords_at_bangkok():
    """Bangkok at zoom 19 should give a sensible tile coord."""
    fetcher = TileFetcher()
    x, y = fetcher._tile_coords(13.7563, 100.5018, 19)
    # At zoom 19, n=2^19=524288. Bangkok is ~78% east, ~46% south.
    assert 0 < x < 524288
    assert 0 < y < 524288


@pytest.mark.asyncio
async def test_fetch_esri_calls_correct_url(monkeypatch):
    """Esri fetch should hit the World_Imagery endpoint with z/y/x ordering."""
    fetcher = TileFetcher()

    captured: dict = {}

    class FakeResponse:
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG

        def raise_for_status(self):
            pass

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, url, **kwargs):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr("app.inference.tile_fetch.httpx.AsyncClient", lambda: FakeClient())

    await fetcher._fetch_esri(zoom=19, x=100, y=200)
    assert "World_Imagery" in captured["url"]
    # Esri uses z/y/x not z/x/y — verify
    assert "/19/200/100" in captured["url"]


@pytest.mark.asyncio
async def test_fetch_mapbox_requires_token():
    fetcher = TileFetcher(mapbox_token="")
    with pytest.raises(ValueError, match="MAPBOX_TOKEN"):
        await fetcher._fetch_mapbox(zoom=19, x=100, y=200)


# ═══════════════════════════════════════════════════════════════
# /quote/sync integration test
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _fake_tile_image(size: int = 256) -> np.ndarray:
    """Generate a fake satellite tile (gray RGB)."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


@pytest.mark.asyncio
async def test_sync_quote_requires_address_or_coords(client):
    async with client as c:
        resp = await c.post("/quote/sync", json={})
    assert resp.status_code == 400
    assert "address" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_sync_quote_end_to_end_thai_address(client, monkeypatch):
    """Full /quote/sync run for a Thai address with all externals mocked.

    Verifies:
    - geocode is called and returns a Thai coord
    - tile fetcher returns a fake image
    - FM pipeline runs and returns a CorrosionResult
    - quote is computed in THB
    """
    monkeypatch.setenv("REGION", "TH")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake-test-key")

    # Mock geocode → Bangkok center
    async def fake_geocode(address):
        return (13.7563, 100.5018)

    # Mock tile fetch → fake gray image with bounds
    async def fake_fetch_tiles(job):
        return _fake_tile_image(256), 0.30, (13.74, 100.49, 13.76, 100.51)

    # Mock the FM pipeline analyze → return a synthetic result
    from app.inference.pipeline import CorrosionResult

    fake_result = CorrosionResult(
        roof_area_m2=180.0,
        corroded_area_m2=27.0,
        corrosion_percent=15.0,
        severity="light",
        confidence=0.82,
        roof_mask=np.ones((256, 256), dtype=bool),
        corrosion_mask=np.zeros((256, 256), dtype=bool),
        gsd=0.30,
        roof_model_version="osm:way/123",
        corrosion_model_version="nim/llama-3.2-90b-vision-instruct",
    )

    async def fake_analyze(self, **kwargs):
        return fake_result

    # Mock supabase failure so quote engine uses region defaults (THB)
    def raise_db_error():
        raise Exception("no supabase in test")

    monkeypatch.setattr("app.inference.worker.geocode_address", fake_geocode)
    monkeypatch.setattr("app.routes.quote.geocode_address", fake_geocode, raising=False)
    monkeypatch.setattr("app.inference.worker.fetch_tiles_for_job", fake_fetch_tiles)
    monkeypatch.setattr("app.routes.quote.fetch_tiles_for_job", fake_fetch_tiles, raising=False)
    monkeypatch.setattr(
        "app.inference.pipeline_fm.FoundationModelPipeline.analyze",
        fake_analyze,
    )
    monkeypatch.setattr("app.quote_engine.get_supabase", raise_db_error)

    payload = {"address": "88 Bangna-Trad Rd, Samut Prakan 10540, Thailand"}

    async with client as c:
        resp = await c.post("/quote/sync", json=payload)

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Job metadata
    assert data["status"] == "completed"
    assert data["lat"] == pytest.approx(13.7563, rel=0.01)
    assert data["lng"] == pytest.approx(100.5018, rel=0.01)
    assert data["gsd_m"] == pytest.approx(0.30, abs=0.01)

    # Assessment
    assert data["assessment"]["roof_area_m2"] == 180.0
    assert data["assessment"]["severity"] == "light"
    assert 0 < data["assessment"]["confidence"] <= 1.0

    # Quote in THB (Thailand region)
    assert data["quote"]["currency"] == "THB"
    assert data["quote"]["total_amount"] > 0
    line_items = data["quote"]["line_items"]
    assert len(line_items) >= 2  # inspection + at least one service line
    inspection = next(li for li in line_items if "inspection" in li["description"].lower())
    # THB inspection fee is ~2500
    assert inspection["unit_price"] > 1000


@pytest.mark.asyncio
async def test_sync_quote_geocoding_failure_returns_400(client, monkeypatch):
    """If geocoding fails, return 400 with a clear error."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake-test-key")

    async def fake_geocode_fail(address):
        raise ValueError("No in-region match")

    # Must patch the source module — the sync endpoint uses a lazy import:
    #   from app.inference.worker import geocode_address
    monkeypatch.setattr("app.inference.worker.geocode_address", fake_geocode_fail)

    async with client as c:
        resp = await c.post("/quote/sync", json={"address": "Atlantis"})
    assert resp.status_code == 400
    assert "Geocoding" in resp.json()["detail"]
