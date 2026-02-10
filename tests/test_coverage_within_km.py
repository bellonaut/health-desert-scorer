import importlib.util

import pytest


if importlib.util.find_spec("geopandas") is None:
    pytest.skip("geopandas not installed", allow_module_level=True)

import geopandas as gpd  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from src.data.spatial_ops import coverage_within_km  # noqa: E402


def _make_lga_box(x0, y0, x1, y1, name, uid):
    return gpd.GeoDataFrame(
        {"lga_id": [uid], "lga_name": [name]},
        geometry=[box(x0, y0, x1, y1)],
        crs="EPSG:4326",
    )


def _make_facilities(points):
    return gpd.GeoDataFrame({"geometry": points}, geometry="geometry", crs="EPSG:4326")


class TestCoverageWithinKmBounds:
    def test_overlapping_buffers_clamped_to_100(self):
        lgas = _make_lga_box(0, 0, 1, 1, "A", 1)
        points = [
            Point(0.5 + (i % 5) * 0.0001, 0.5 + (i // 5) * 0.0001) for i in range(50)
        ]
        facilities = _make_facilities(points)

        result = coverage_within_km(lgas, facilities, km=5)
        pct = result.loc[0, "area_covered_pct"]
        assert 0.0 <= pct <= 100.0


class TestCoverageWithinKmSemantics:
    def test_empty_facilities_yield_zero(self):
        lgas = _make_lga_box(0, 0, 1, 1, "A", 1)
        facilities = _make_facilities([])

        result = coverage_within_km(lgas, facilities, km=5)
        assert result.loc[0, "area_covered_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_far_facility_has_near_zero_coverage(self):
        lgas = _make_lga_box(0, 0, 1, 1, "A", 1)
        facilities = _make_facilities([Point(5, 5)])  # ~780 km away

        result = coverage_within_km(lgas, facilities, km=5)
        assert result.loc[0, "area_covered_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_centroid_facility_produces_positive_coverage(self):
        lgas = _make_lga_box(0, 0, 1, 1, "A", 1)
        facilities = _make_facilities([Point(0.5, 0.5)])

        result = coverage_within_km(lgas, facilities, km=5)
        assert result.loc[0, "area_covered_pct"] > 0

    def test_more_facilities_monotonic_non_decreasing(self):
        lgas = _make_lga_box(0, 0, 1, 1, "A", 1)
        base_facilities = _make_facilities([Point(0.5, 0.5)])
        expanded_facilities = _make_facilities(
            [Point(0.5, 0.5), Point(0.1, 0.1), Point(0.9, 0.9)]
        )

        base_pct = coverage_within_km(lgas, base_facilities, km=5).loc[0, "area_covered_pct"]
        expanded_pct = coverage_within_km(lgas, expanded_facilities, km=5).loc[0, "area_covered_pct"]
        assert expanded_pct >= base_pct

    def test_facility_only_impacts_overlapping_lga(self):
        lgas = gpd.GeoDataFrame(
            {
                "lga_id": [1, 2],
                "lga_name": ["A", "B"],
            },
            geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
            crs="EPSG:4326",
        )
        facilities = _make_facilities([Point(0.5, 0.5)])

        result = coverage_within_km(lgas, facilities, km=5)
        pct_a = result.loc[result["lga_name"] == "A", "area_covered_pct"].iloc[0]
        pct_b = result.loc[result["lga_name"] == "B", "area_covered_pct"].iloc[0]

        assert pct_a > 0
        assert pct_b == pytest.approx(0.0, abs=1e-6)


class TestCoverageWithinKmRowCount:
    def test_row_count_matches_lgas(self):
        lgas = gpd.GeoDataFrame(
            {
                "lga_id": [1, 2, 3],
                "lga_name": ["A", "B", "C"],
            },
            geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)],
            crs="EPSG:4326",
        )
        facilities = _make_facilities([Point(0.5, 0.5), Point(2.5, 2.5)])

        result = coverage_within_km(lgas, facilities, km=5)
        assert len(result) == len(lgas)
