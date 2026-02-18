"""HTML embedding bridge for the Health Desert app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
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
CSS_PATH = APP_DIR / "health_desert_ui.css"
JS_PATH = APP_DIR / "health_desert_ui.js"


def _json_default(obj: Any) -> Any:
    """Exhaustive JSON serialization fallback for numpy/pandas/shapely types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if not pd.isnull(obj) else None
    if isinstance(obj, np.datetime64):
        ts = pd.Timestamp(obj)
        return ts.isoformat() if not pd.isnull(ts) else None
    try:
        from shapely.geometry.base import BaseGeometry

        if isinstance(obj, BaseGeometry):
            return None
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Last resort: stringify instead of crashing the entire app.
    try:
        return str(obj)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and (value != value):  # NaN check
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _year_key(value: Any) -> int | str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return str(value)


def _build_shap_lookup(shap_df: pd.DataFrame | None) -> dict[tuple[str, int | str | None], dict[str, float]]:
    if shap_df is None or shap_df.empty or "lga_name" not in shap_df.columns:
        return {}

    feature_cols = [c for c in shap_df.columns if c not in {"lga_name", "year", "is_synthetic", "shap_importance"}]
    if not feature_cols:
        return {}

    numeric = shap_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    lookup: dict[tuple[str, int | str | None], dict[str, float]] = {}
    for idx, row in shap_df.iterrows():
        lga_name = str(row.get("lga_name"))
        if not lga_name:
            continue

        shap_map: dict[str, float] = {}
        for col in feature_cols:
            val = numeric.at[idx, col]
            if pd.notna(val):
                shap_map[col] = float(val)
        if not shap_map:
            continue

        y_key = _year_key(row.get("year")) if "year" in shap_df.columns else None
        if y_key is not None:
            lookup[(lga_name, y_key)] = shap_map
        lookup.setdefault((lga_name, None), shap_map)

    return lookup


def _coerce(value: Any) -> Any:
    """Convert any value to a JSON-safe Python native type."""
    if value is None:
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if not pd.isnull(value) else None
    if isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
        return ts.isoformat() if not pd.isnull(ts) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    # numpy.object_ and opaque numpy scalar wrappers.
    if hasattr(value, "item"):
        try:
            return _coerce(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        return str(value)
    except Exception:
        return None


def _find_unserializable(obj: Any, path: str = "root") -> str | None:
    """Walk nested payloads to identify the first problematic field."""
    try:
        json.dumps(obj, default=_json_default)
        return None
    except Exception:
        if isinstance(obj, dict):
            for key, value in obj.items():
                found = _find_unserializable(value, f"{path}.{key}")
                if found:
                    return found
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                found = _find_unserializable(value, f"{path}[{idx}]")
                if found:
                    return found

        preview = repr(obj)
        if len(preview) > 80:
            preview = preview[:80]
        return f"{path} = {type(obj).__name__}({preview})"


def _records_from_geo(
    filtered_df,
    include_shap: bool = False,
    shap_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    shap_lookup = _build_shap_lookup(shap_df) if include_shap else {}
    records: list[dict[str, Any]] = []
    for row in filtered_df.itertuples():
        lga_name = _coerce(getattr(row, "lga_name"))
        year_value = _coerce(getattr(row, "year", None))
        year_key = _year_key(year_value)
        rec: dict[str, Any] = {
            "id": str(getattr(row, "lga_uid")),
            "name": lga_name,
            "state": _coerce(getattr(row, "state_name")),
            "risk": _safe_float(getattr(row, "risk_score", None)),
            "risk_total": _safe_float(getattr(row, "risk_score_total", None)),
            "fac": _safe_float(getattr(row, "facilities_per_10k", None)),
            "dist": _safe_float(getattr(row, "avg_distance_km", None)),
            "u5mr": _safe_float(getattr(row, "u5mr_mean", None)),
            "pop": _safe_float(getattr(row, "population", None)),
            "cov": _safe_float(getattr(row, "coverage_5km", None)),
            "towers": _safe_float(getattr(row, "towers_per_10k", None)),
            "density": _safe_float(getattr(row, "population_density", None)),
            "year": year_value,
            "confidence_pct": _safe_float(getattr(row, "confidence_pct", None)),
            "confidence_reason_codes": _coerce(getattr(row, "confidence_reason_codes", None)),
            "primary_barriers": _coerce(getattr(row, "primary_barriers", None)),
            "recommendation": _coerce(getattr(row, "recommendation", None)),
        }
        if include_shap:
            by_year = shap_lookup.get((str(lga_name), year_key))
            rec["shap"] = by_year if by_year is not None else shap_lookup.get((str(lga_name), None))
        records.append(rec)
    return records


def build_payload(geo_df, shap_df, session_state: Mapping[str, Any]) -> dict[str, Any]:
    year = session_state.get("hd_year") or latest_year(geo_df)
    state_filter = session_state.get("hd_state_filter", "All Nigeria")
    focus = session_state.get("hd_focus", "All risk")
    depth = int(session_state.get("hd_depth", 0) or 0)
    is_mobile = bool(session_state.get("hd_is_mobile"))
    selected_lga = session_state.get("hd_selected_lga")
    compare_lgas = [str(uid) for uid in session_state.get("hd_compare_lgas", [])]

    filtered = filter_geo(geo_df, state_filter=state_filter, year=year)
    include_shap = depth >= 1 and (str(year).lower() != "both")
    lga_records = _records_from_geo(
        filtered,
        include_shap=include_shap,
        shap_df=shap_df if include_shap else None,
    )

    hotspots = get_ranked_hotspots(geo_df, focus, state_filter=state_filter, year=year, limit=12)
    shap_allowed = depth >= 1 and (str(year).lower() != "both")
    selected_detail = get_lga_detail(geo_df, shap_df if shap_allowed else None, selected_lga, year=year) if selected_lga else None

    risk_norm = normalize_for_choropleth(filtered, "risk_score")
    map_values = [
        {"id": rec["id"], "risk_norm": risk_norm[idx], "risk": rec["risk"]}
        for idx, rec in enumerate(lga_records)
    ]

    if is_mobile:
        geojson_columns = ("lga_uid",)
    else:
        geojson_columns = ("lga_uid", "lga_name", "state_name", "risk_score")

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
            "boundary_resolution": geo_df.attrs.get("boundary_resolution"),
            "data_source_mode": geo_df.attrs.get("data_source_mode"),
            "data_last_updated": geo_df.attrs.get("data_last_updated"),
            "model_version": geo_df.attrs.get("model_version"),
        },
        "states": get_states(geo_df),
        "lgas": lga_records,
        "hotspots": hotspots,
        "selected": selected_detail,
        "map": {
            "geojson": get_lgas_geojson(
                geo_df,
                state_filter=state_filter,
                year=year,
                columns=geojson_columns,
            ),
            "choropleth": map_values,
        },
    }
    return payload


def inject_data_to_html(html_path: Path, data: dict[str, Any]) -> str:
    try:
        json_str = json.dumps(data, default=_json_default)
    except Exception as exc:
        bad_field = _find_unserializable(data) or "unknown"
        st.error(
            f"**Serialization error** - {exc}\n\n"
            f"**Offending field:** `{bad_field}`\n\n"
            "Check bridge.py `_records_from_geo` for uncoerced types."
        )
        st.stop()

    html = html_path.read_text(encoding="utf-8")
    injection = f"<script>window.__INITIAL_DATA__ = {json_str};</script>"

    css_text = CSS_PATH.read_text(encoding="utf-8")
    js_text = JS_PATH.read_text(encoding="utf-8")

    if "<!-- APP_STYLE -->" in html:
        html = html.replace("<!-- APP_STYLE -->", f"<style>\n{css_text}\n</style>")
    if "<!-- APP_SCRIPT -->" in html:
        html = html.replace("<!-- APP_SCRIPT -->", f"<script>\n{js_text}\n</script>")

    if "<!-- DATA_INJECTION -->" in html:
        html = html.replace("<!-- DATA_INJECTION -->", injection)
        return html
    if "</head>" in html:
        return html.replace("</head>", f"{injection}\n</head>", 1)
    return injection + html


def render_embedded_app(
    geo_df,
    shap_df,
    session_state: Mapping[str, Any],
    html_path: Path = HTML_PATH,
    height: int = 10000,
) -> None:
    payload = build_payload(geo_df, shap_df, session_state)
    injected = inject_data_to_html(html_path, payload)
    st.components.v1.html(injected, height=height, scrolling=False)
