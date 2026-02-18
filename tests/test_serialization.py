"""Serialization safety tests for embedded payload generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("geopandas") is None:
    pytest.skip("geopandas not installed", allow_module_level=True)

import geopandas as gpd  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bridge import _json_default, build_payload  # noqa: E402


class _Opaque:
    def __str__(self) -> str:
        return "opaque-value"


def test_numpy_types() -> None:
    cases = [
        (np.int64(5), 5),
        (np.float64(3.14), 3.14),
        (np.float64(np.nan), None),
        (np.bool_(True), True),
        (np.array([1, 2]), [1, 2]),
        (np.object_("opaque"), "opaque"),
        (pd.NA, None),
        (pd.NaT, None),
        (pd.Timestamp("2018-01-01"), "2018-01-01T00:00:00"),
        (_Opaque(), "opaque-value"),
    ]
    for obj, expected in cases:
        assert _json_default(obj) == expected, f"Failed for {type(obj)}: {obj}"


@pytest.fixture
def sample_geo_df() -> gpd.GeoDataFrame:
    poly = Polygon([(3.0, 6.0), (3.2, 6.0), (3.2, 6.2), (3.0, 6.2)])
    df = gpd.GeoDataFrame(
        {
            "lga_uid": [np.int64(101)],
            "lga_name": ["Test LGA"],
            "state_name": ["Test State"],
            "risk_score": [np.float64(0.72)],
            "risk_score_total": [np.float64(7.2)],
            "facilities_per_10k": [np.float64(0.4)],
            "avg_distance_km": [np.float64(9.1)],
            "u5mr_mean": [np.float64(156.0)],
            "population": [np.float64(250000)],
            "coverage_5km": [np.float64(21.0)],
            "towers_per_10k": [np.float64(0.9)],
            "population_density": [np.float64(450.0)],
            "year": [np.int64(2018)],
            "confidence_pct": [np.float64(81.0)],
            "confidence_reason_codes": [set(["model_v1", "complete_inputs"])],
            "primary_barriers": [set(["Low facility density"])],
            "recommendation": [b"Deploy mobile clinics"],
        },
        geometry=[poly],
        crs="EPSG:4326",
    )
    df.attrs["boundary_resolution"] = "high"
    df.attrs["data_source_mode"] = "gold_first"
    df.attrs["data_last_updated"] = "2026-02-18"
    df.attrs["model_version"] = ["v1.2"]
    return df


@pytest.fixture
def sample_shap_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lga_name": ["Test LGA"],
            "year": [2018],
            "feature_a": [0.31],
            "feature_b": [-0.22],
        }
    )


@pytest.fixture
def sample_session() -> dict[str, object]:
    return {
        "hd_year": 2018,
        "hd_state_filter": "All Nigeria",
        "hd_focus": "All risk",
        "hd_depth": 2,
        "hd_selected_lga": "101",
        "hd_compare_lgas": [np.int64(101)],
        "hd_is_mobile": True,
    }


def test_full_payload_serializable(sample_geo_df, sample_shap_df, sample_session) -> None:
    """End-to-end: build_payload must produce JSON-serializable output."""
    payload = build_payload(sample_geo_df, sample_shap_df, sample_session)
    json.dumps(payload, default=_json_default)
