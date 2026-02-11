"""HTML embedding bridge for the Health Desert app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from data_api import (
    FOCUS_COLUMN,
    filter_geo,
    get_lga_detail,
    get_lgas_geojson,
    get_ranked_hotspots,
    get_states,
    latest_year,
    normalize_for_choropleth,
)

APP_DIR = Path(__file__).resolve().parent
HTML_PATH = APP_DIR / "health_desert_ui.html"


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and (value != value):  # NaN check
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _records_from_geo(filtered_df, include_shap: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in filtered_df.itertuples():
        rec: dict[str, Any] = {
            "id": str(getattr(row, "lga_uid")),
            "name": getattr(row, "lga_name"),
            "state": getattr(row, "state_name"),
            "risk": _safe_float(getattr(row, "risk_score", None)),
            "fac": _safe_float(getattr(row, "facilities_per_10k", None)),
            "dist": _safe_float(getattr(row, "avg_distance_km", None)),
            "u5mr": _safe_float(getattr(row, "u5mr_mean", None)),
            "pop": _safe_float(getattr(row, "population", None)),
            "cov": _safe_float(getattr(row, "coverage_5km", None)),
            "towers": _safe_float(getattr(row, "towers_per_10k", None)),
            "density": _safe_float(getattr(row, "population_density", None)),
            "year": getattr(row, "year", None),
        }
        if include_shap:
            rec["shap"] = None  # populated per-selection to keep payload light
        records.append(rec)
    return records


def build_payload(geo_df, shap_df, session_state: Mapping[str, Any]) -> dict[str, Any]:
    year = session_state.get("hd_year") or latest_year(geo_df)
    state_filter = session_state.get("hd_state_filter", "All Nigeria")
    focus = session_state.get("hd_focus", "All risk")
    depth = int(session_state.get("hd_depth", 0) or 0)
    selected_lga = session_state.get("hd_selected_lga")
    compare_lgas = [str(uid) for uid in session_state.get("hd_compare_lgas", [])]

    filtered = filter_geo(geo_df, state_filter=state_filter, year=year)
    lga_records = _records_from_geo(filtered, include_shap=depth >= 2)

    hotspots = get_ranked_hotspots(geo_df, focus, state_filter=state_filter, year=year, limit=12)
    shap_allowed = depth >= 2 and (str(year).lower() != "both")
    selected_detail = get_lga_detail(geo_df, shap_df if shap_allowed else None, selected_lga, year=year) if selected_lga else None

    risk_norm = normalize_for_choropleth(filtered, "risk_score")
    map_values = [
        {"id": rec["id"], "risk_norm": risk_norm[idx], "risk": rec["risk"]}
        for idx, rec in enumerate(lga_records)
    ]

    payload: dict[str, Any] = {
        "meta": {
            "state_filter": state_filter,
            "depth": depth,
            "focus": focus,
            "year": year,
            "selected_lga": selected_lga,
            "compare_lgas": compare_lgas,
            "lga_count": int(filtered["lga_uid"].nunique()) if "lga_uid" in filtered.columns else len(filtered),
            "focus_column": FOCUS_COLUMN.get(focus, "risk_score"),
        },
        "states": get_states(geo_df),
        "lgas": lga_records,
        "hotspots": hotspots,
        "selected": selected_detail,
        "map": {
            "geojson": get_lgas_geojson(geo_df, state_filter=state_filter, year=year),
            "choropleth": map_values,
        },
    }
    return payload


def inject_data_to_html(html_path: Path, data: dict[str, Any]) -> str:
    html = html_path.read_text(encoding="utf-8")
    injection = f"<script>window.__INITIAL_DATA__ = {json.dumps(data, default=_json_default)};</script>"
    if "<!-- DATA_INJECTION -->" in html:
        return html.replace("<!-- DATA_INJECTION -->", injection)
    if "</head>" in html:
        return html.replace("</head>", f"{injection}\n</head>", 1)
    return injection + html


def render_embedded_app(
    geo_df,
    shap_df,
    session_state: Mapping[str, Any],
    html_path: Path = HTML_PATH,
    height: int = 1100,
) -> None:
    payload = build_payload(geo_df, shap_df, session_state)
    injected = inject_data_to_html(html_path, payload)
    st.components.v1.html(injected, height=height, scrolling=False)
