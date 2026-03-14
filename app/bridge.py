"""HTML embedding bridge for the Health Desert app."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from data_api import (
    FOCUS_COLUMN,
    _worst_driver,
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
ASSETS_DIR = APP_DIR / "assets"
VENDORED_CSS = ("leaflet.css", "leaflet.fullscreen.css")
VENDORED_JS = ("leaflet.js", "leaflet.fullscreen.js")


def _sanitize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Replace float NaN/Inf with None so browser JSON parsing stays valid."""
    out: dict[str, Any] = {}
    for key, value in rec.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


def _sanitize_json_tree(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None in nested payloads."""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_json_tree(item) for item in obj.tolist()]
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if not pd.isnull(obj) else None
    if isinstance(obj, np.datetime64):
        ts = pd.Timestamp(obj)
        return ts.isoformat() if not pd.isnull(ts) else None
    if isinstance(obj, dict):
        return {str(key): _sanitize_json_tree(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_sanitize_json_tree(value) for value in obj]
    return obj


class _NaNSafeEncoder(json.JSONEncoder):
    def encode(self, obj: Any) -> str:
        return super().encode(_sanitize_json_tree(obj))


def _inline_css_urls(css_text: str, css_path: Path) -> str:
    """Rewrite relative CSS url(...) references to data URIs for iframe-safe embeds."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip().strip("\"'")
        if not raw or raw.startswith(("data:", "http://", "https://", "#")):
            return match.group(0)

        asset_path = (css_path.parent / raw).resolve()
        if not asset_path.exists():
            asset_path = (ASSETS_DIR / Path(raw).name).resolve()
        if not asset_path.exists():
            return match.group(0)

        mime = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f"url(data:{mime};base64,{encoded})"

    return re.sub(r"url\(([^)]+)\)", replace, css_text)


def _read_vendored_css() -> str:
    bundled: list[str] = []
    for name in VENDORED_CSS:
        css_path = ASSETS_DIR / name
        css_text = css_path.read_text(encoding="utf-8")
        bundled.append(_inline_css_urls(css_text, css_path))
    return "\n".join(bundled)


def _read_vendored_js() -> str:
    return "".join(f"<script>\n{(ASSETS_DIR / name).read_text(encoding='utf-8')}\n</script>" for name in VENDORED_JS)


def _json_default(obj: Any) -> Any:
    """Exhaustive JSON serialization fallback for numpy/pandas/shapely types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
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
        n = float(value)
        if math.isnan(n) or math.isinf(n):
            return None
        return n
    except (TypeError, ValueError):
        return None


def _cap_confidence(value: Any, cap: float = 90.0) -> float | None:
    """Mirror of data_api._cap_confidence. Cap confidence to honest band."""
    n = _safe_float(value)
    if n is None:
        return None
    return min(n, cap)


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
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
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


def _worst_driver_from_row(row: Mapping[str, Any]) -> str:
    return _worst_driver(row)


def _records_from_geo(
    filtered_df,
    include_shap: bool = False,
    shap_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    shap_lookup = _build_shap_lookup(shap_df) if include_shap else {}
    # Replace all NaN with None so downstream serialization is safe.
    filtered_df = filtered_df.replace({np.nan: None})
    records: list[dict[str, Any]] = []
    for row in filtered_df.itertuples():
        lga_name = _coerce(getattr(row, "lga_name"))
        year_value = _coerce(getattr(row, "year", None))
        year_key = _year_key(year_value)
        driver_source = {
            "risk_score_mortality": _coerce(getattr(row, "risk_score_mortality", None)),
            "risk_score_facility_access": _coerce(getattr(row, "risk_score_facility_access", None)),
            "risk_score_connectivity": _coerce(getattr(row, "risk_score_connectivity", None)),
            "risk_score_access_60min": _coerce(getattr(row, "risk_score_access_60min", None)),
            "u5_mortality_rate": _coerce(getattr(row, "u5_mortality_rate", None)),
            "u5mr_mean": _coerce(getattr(row, "u5mr_mean", None)),
            "facilities_per_10k": _coerce(getattr(row, "facilities_per_10k", None)),
            "coverage_5km": _coerce(getattr(row, "coverage_5km", None)),
            "connectivity_score": _coerce(getattr(row, "connectivity_score", None)),
            "towers_per_10k": _coerce(getattr(row, "towers_per_10k", None)),
            "pop_pct_60min": _coerce(getattr(row, "pop_pct_60min", None)),
        }
        rec: dict[str, Any] = {
            "id": str(getattr(row, "lga_uid")),
            "lga_id": str(getattr(row, "lga_uid")),
            "name": lga_name,
            "lga_name": lga_name,
            "state": _coerce(getattr(row, "state_name")),
            "state_name": _coerce(getattr(row, "state_name")),
            "risk": _safe_float(getattr(row, "risk_score", None)),
            "risk_score": _safe_float(getattr(row, "risk_score_total", None)),
            "risk_total": _safe_float(getattr(row, "risk_score_total", None)),
            "fac": _safe_float(getattr(row, "facilities_per_10k", None)),
            "dist": _safe_float(getattr(row, "avg_distance_km", None)),
            "u5mr": _safe_float(getattr(row, "u5mr_mean", None)),
            "pop": _safe_float(getattr(row, "population", None)),
            "cov": _safe_float(getattr(row, "coverage_5km", None)),
            "pop_pct_60min": _safe_float(getattr(row, "pop_pct_60min", None)),
            "towers": _safe_float(getattr(row, "towers_per_10k", None)),
            "density": _safe_float(getattr(row, "population_density", None)),
            "year": year_value,
            "confidence_pct": _cap_confidence(getattr(row, "confidence_pct", None)),
            "confidence_reason_codes": _coerce(getattr(row, "confidence_reason_codes", None)),
            "primary_barriers": _coerce(getattr(row, "primary_barriers", None)),
            "recommendation": _coerce(getattr(row, "recommendation", None)),
        }
        rec["worst_driver"] = _worst_driver_from_row(driver_source)
        if include_shap:
            by_year = shap_lookup.get((str(lga_name), year_key))
            rec["shap"] = by_year if by_year is not None else shap_lookup.get((str(lga_name), None))
        records.append(_sanitize_record(rec))
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
    geojson_source = filtered.copy()
    if "worst_driver" not in geojson_source.columns:
        geojson_source["worst_driver"] = geojson_source.apply(
            lambda row: _worst_driver_from_row(row.to_dict()),
            axis=1,
        )
    # Always include SHAP for single-year views so client-side depth toggles
    # don't temporarily render stale records without attribution.
    include_shap = str(year).lower() != "both"
    lga_records = _records_from_geo(
        filtered,
        include_shap=include_shap,
        shap_df=shap_df if include_shap else None,
    )

    hotspots = get_ranked_hotspots(geo_df, focus, state_filter=state_filter, year=year, limit=12)
    shap_allowed = str(year).lower() != "both"
    selected_detail = get_lga_detail(geo_df, shap_df if shap_allowed else None, selected_lga, year=year) if selected_lga else None

    risk_norm = normalize_for_choropleth(filtered, "risk_score")
    map_values = [
        _sanitize_record({"id": rec["id"], "risk_norm": risk_norm[idx], "risk": rec["risk"]})
        for idx, rec in enumerate(lga_records)
    ]

    geojson_columns = ("lga_uid", "lga_name", "state_name", "risk_score", "risk_score_total", "worst_driver")

    payload: dict[str, Any] = {
        "meta": {
            "state_filter": state_filter,
            "depth": depth,
            "focus": focus,
            "year": year,
            "selected_lga": selected_lga,
            "compare_lgas": compare_lgas,
            "parent_app_path": session_state.get("hd_parent_app_path"),
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
                geojson_source,
                state_filter=state_filter,
                year=year,
                columns=geojson_columns,
            ),
            "choropleth": map_values,
        },
    }
    return payload


def inject_data_to_html(html_path: Path, data: dict[str, Any]) -> str:
    safe_data = _sanitize_json_tree(data)
    try:
        json_str = json.dumps(safe_data, cls=_NaNSafeEncoder, default=_json_default, allow_nan=False)
    except Exception as exc:
        bad_field = _find_unserializable(safe_data) or "unknown"
        st.error(
            f"**Serialization error** - {exc}\n\n"
            f"**Offending field:** `{bad_field}`\n\n"
            "Check bridge.py `_records_from_geo` for uncoerced types."
        )
        st.stop()

    html = html_path.read_text(encoding="utf-8")
    bundled_css = _read_vendored_css()
    bundled_js = _read_vendored_js()
    injection = f"<script>window.__INITIAL_DATA__ = {json_str};</script>"

    css_text = CSS_PATH.read_text(encoding="utf-8")
    js_text = JS_PATH.read_text(encoding="utf-8")

    if "<!-- APP_STYLE -->" in html:
        html = html.replace("<!-- APP_STYLE -->", f"<style>\n{bundled_css}\n{css_text}\n</style>")
    if "<!-- APP_SCRIPT -->" in html:
        html = html.replace("<!-- APP_SCRIPT -->", f"{bundled_js}\n<script>\n{js_text}\n</script>")

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
) -> str:
    payload = build_payload(geo_df, shap_df, session_state)
    return inject_data_to_html(html_path, payload)
