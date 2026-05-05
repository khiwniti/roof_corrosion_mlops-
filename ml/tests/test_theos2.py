"""Tests for ml/ingestion/theos2.py"""

from shapely.geometry import Polygon

from ingestion.theos2 import estimate_cost, search_catalog


def test_estimate_cost_placeholder():
    polygon_geojson = Polygon([(100.0, 13.0), (100.01, 13.0), (100.01, 13.01), (100.0, 13.01), (100.0, 13.0)]).__geo_interface__
    result = estimate_cost(polygon_geojson)

    assert result["constellation"] == "THEOS-2"
    assert result["rate_thb_per_km2"] is None
    assert result["estimated_cost_thb"] is None
    assert "contact" in result
    assert "gistda.or.th" in result["contact"]


def test_search_catalog_stub():
    polygon_geojson = Polygon([(100.0, 13.0), (100.01, 13.0), (100.01, 13.01), (100.0, 13.01), (100.0, 13.0)]).__geo_interface__
    # Without auth, should return empty list
    result = search_catalog(polygon_geojson)
    assert result == []
