"""Unit tests for spatial operations."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from src.data.spatial_ops import assign_points_to_lga, coverage_within_km


def _make_lga_square() -> gpd.GeoDataFrame:
    polygon = Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])
    return gpd.GeoDataFrame(
        {"lga_id": [0], "lga_name": ["Test LGA"]},
        geometry=[polygon],
        crs="EPSG:4326",
    )


def _make_facility_point() -> gpd.GeoDataFrame:
    point = Point(0.05, 0.05)
    return gpd.GeoDataFrame(
        {"facility_name": ["Test Facility"], "facility_type": ["clinic"]},
        geometry=[point],
        crs="EPSG:4326",
    )


def test_assign_points_requires_crs() -> None:
    points = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    lgas = _make_lga_square()
    with pytest.raises(ValueError, match="missing CRS"):
        assign_points_to_lga(points, lgas)


def test_coverage_within_km_outputs() -> None:
    lgas = _make_lga_square()
    facilities = _make_facility_point()
    coverage = coverage_within_km(lgas, facilities, km=1.0)
    assert list(coverage.columns) == ["lga_id", "lga_name", "area_covered_pct"]
    assert coverage.shape[0] == 1
    pct = coverage.loc[0, "area_covered_pct"]
    assert pd.notna(pct)
    assert 0 <= pct <= 100
