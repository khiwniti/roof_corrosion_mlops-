"""Tests for ml/ingestion/pleiades.py"""

import pytest
from shapely.geometry import Polygon

from ingestion.pleiades import estimate_cost


def test_estimate_cost_phhr():
    polygon_geojson = Polygon([(100.0, 13.0), (100.01, 13.0), (100.01, 13.01), (100.0, 13.01), (100.0, 13.0)]).__geo_interface__
    result = estimate_cost(polygon_geojson, constellation="PHR")

    assert result["constellation"] == "PHR"
    assert result["rate_eur_per_km2"] == 3.80
    assert result["estimated_cost_eur"] > 0
    assert result["estimated_cost_thb"] > 0
    assert "aggregation_note" in result


def test_estimate_cost_spot():
    polygon_geojson = Polygon([(100.0, 13.0), (100.01, 13.0), (100.01, 13.01), (100.0, 13.01), (100.0, 13.0)]).__geo_interface__
    result = estimate_cost(polygon_geojson, constellation="SP6")

    assert result["constellation"] == "SP6"
    assert result["rate_eur_per_km2"] == 2.50


def test_estimate_cost_default():
    polygon_geojson = Polygon([(100.0, 13.0), (100.01, 13.0), (100.01, 13.01), (100.0, 13.01), (100.0, 13.0)]).__geo_interface__
    result = estimate_cost(polygon_geojson)

    assert result["constellation"] == "PHR"
    assert result["rate_eur_per_km2"] == 3.80
