"""Tests for Thailand region scoping (default REGION=TH)."""

from __future__ import annotations

import os

import pytest

from app.region import (
    THAILAND_BBOX,
    THAI_CITY_CENTERS,
    THAILAND_PROFILE,
    USA_PROFILE,
    REGION_PROFILES,
    get_active_region,
    is_in_region,
)


# ═══════════════════════════════════════════════════════════════
# Region profile basics
# ═══════════════════════════════════════════════════════════════

def test_thailand_profile_defaults():
    assert THAILAND_PROFILE.code == "TH"
    assert THAILAND_PROFILE.currency == "THB"
    assert THAILAND_PROFILE.country_code == "th"
    assert "monsoon" in THAILAND_PROFILE.roof_context_hint.lower()
    assert "metal" in THAILAND_PROFILE.roof_context_hint.lower()


def test_thailand_pricing_is_thb_scaled():
    # THB prices should be 15-30x USD prices (rough THB:USD ratio is ~35:1,
    # but Thai roof labor is cheaper, so ~15-25x is normal)
    th = THAILAND_PROFILE.price_per_m2
    us = USA_PROFILE.price_per_m2
    assert th["replacement"] / us["replacement"] > 10
    assert th["replacement"] / us["replacement"] < 30
    assert th["coating"] > 100  # THB
    assert us["coating"] < 30   # USD


def test_active_region_defaults_to_thailand(monkeypatch):
    monkeypatch.delenv("REGION", raising=False)
    assert get_active_region().code == "TH"


def test_active_region_respects_env(monkeypatch):
    monkeypatch.setenv("REGION", "US")
    assert get_active_region().code == "US"
    monkeypatch.setenv("REGION", "us")
    assert get_active_region().code == "US"  # case-insensitive


def test_active_region_falls_back_to_thailand_for_unknown(monkeypatch):
    monkeypatch.setenv("REGION", "ZZ")
    assert get_active_region().code == "TH"


# ═══════════════════════════════════════════════════════════════
# Bounding-box checks
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("city,coords", THAI_CITY_CENTERS.items())
def test_thai_cities_inside_bbox(city, coords):
    lat, lng = coords
    assert is_in_region(lat, lng, THAILAND_PROFILE), f"{city} should be in TH"


@pytest.mark.parametrize("city,coords", [
    ("singapore", (1.3521, 103.8198)),
    ("kuala_lumpur", (3.1390, 101.6869)),
    ("ho_chi_minh", (10.7626, 106.6602)),
    ("yangon", (16.8409, 96.1735)),
    ("hong_kong", (22.3193, 114.1694)),
    ("nyc", (40.7128, -74.0060)),
    ("london", (51.5074, -0.1278)),
])
def test_non_thai_cities_outside_bbox(city, coords):
    lat, lng = coords
    assert not is_in_region(lat, lng, THAILAND_PROFILE), f"{city} should be outside TH"


def test_global_profile_is_in_region_always_true():
    # USA_PROFILE has no bbox → all coords accepted
    assert is_in_region(40.0, -74.0, USA_PROFILE)
    assert is_in_region(0.0, 0.0, USA_PROFILE)


# ═══════════════════════════════════════════════════════════════
# Integration with NIM prompt builder
# ═══════════════════════════════════════════════════════════════

def test_nim_prompt_includes_thailand_context_by_default(monkeypatch):
    monkeypatch.delenv("REGION", raising=False)
    from app.inference.nim_client import build_user_prompt
    prompt = build_user_prompt(gsd=0.3, address="Bangkok, Thailand")
    assert "Thailand" in prompt
    assert "monsoon" in prompt.lower() or "tropical" in prompt.lower()


def test_nim_prompt_omits_thailand_when_region_us(monkeypatch):
    monkeypatch.setenv("REGION", "US")
    from app.inference.nim_client import build_user_prompt
    prompt = build_user_prompt(gsd=0.3, address="123 Main St")
    assert "Thailand" not in prompt
    # US profile has empty roof_context_hint
    assert "monsoon" not in prompt.lower()


# ═══════════════════════════════════════════════════════════════
# Integration with quote engine
# ═══════════════════════════════════════════════════════════════

def test_quote_uses_thb_in_thailand(monkeypatch):
    """End-to-end: TH region → quote currency is THB and prices are scaled."""
    monkeypatch.delenv("REGION", raising=False)
    from app.quote_engine import compute_quote

    # Mock supabase failure → falls back to region defaults
    monkeypatch.setattr("app.quote_engine.get_supabase", lambda: (_ for _ in ()).throw(Exception()))

    result = compute_quote(
        roof_area_m2=200.0,
        corroded_area_m2=20.0,
        corrosion_percent=10.0,
        severity="light",
        confidence=0.85,
    )

    assert result.currency == "THB"
    # Inspection fee in THB (2500) is much higher than USD (150)
    inspection_line = next(li for li in result.line_items if "inspection" in li["description"].lower())
    assert inspection_line["unit_price"] >= 1000  # definitely THB, not USD


def test_quote_uses_usd_when_region_us(monkeypatch):
    monkeypatch.setenv("REGION", "US")
    from app.quote_engine import compute_quote

    monkeypatch.setattr("app.quote_engine.get_supabase", lambda: (_ for _ in ()).throw(Exception()))

    result = compute_quote(
        roof_area_m2=200.0,
        corroded_area_m2=20.0,
        corrosion_percent=10.0,
        severity="light",
        confidence=0.85,
    )

    assert result.currency == "USD"
    inspection_line = next(li for li in result.line_items if "inspection" in li["description"].lower())
    assert inspection_line["unit_price"] < 200  # USD inspection fee
