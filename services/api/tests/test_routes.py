"""Tests for FastAPI routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health_check(client):
    async with client as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_check(client):
    async with client as c:
        response = await c.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_submit_quote_with_address(client):
    async with client as c:
        response = await c.post(
            "/quote/",
            json={"address": "123 Main St, Springfield, IL"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_quote_with_coords(client):
    async with client as c:
        response = await c.post(
            "/quote/",
            json={"lat": -6.2088, "lng": 106.8456},
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_get_quote_status_unknown_job(client):
    """Unknown job ID returns 'unknown' status (no DB/Redis in test env)."""
    async with client as c:
        response = await c.get("/quote/nonexistent-job-id")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "nonexistent-job-id"
    assert data["status"] == "unknown"


@pytest.mark.asyncio
async def test_submit_feedback(client):
    async with client as c:
        response = await c.post(
            "/feedback/",
            json={"job_id": "test-job", "correct": False, "notes": "Corrosion area too small"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["job_id"] == "test-job"
    assert data["flagged_for_relabeling"] is True


@pytest.mark.asyncio
async def test_submit_feedback_correct(client):
    """Correct feedback should NOT flag for relabeling."""
    async with client as c:
        response = await c.post(
            "/feedback/",
            json={"job_id": "test-job-2", "correct": True},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["flagged_for_relabeling"] is False


@pytest.mark.asyncio
async def test_submit_quote_requires_address_or_coords(client):
    """Quote request without address AND without coords should return 400."""
    async with client as c:
        response = await c.post("/quote/", json={})
    assert response.status_code == 400
