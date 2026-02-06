"""Streamlit UI components."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

APPLE_COLORS = {
    # High-contrast palette tuned for light basemap
    "green": (0, 140, 70),
    "yellow": (255, 190, 0),
    "red": (214, 40, 57),
    "gray": (200, 200, 200),
    "ink": (0, 0, 0),
    "highlight": (0, 190, 255),  # cyan highlight
}

# Tokenless light basemap with minimal visual noise
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def _interpolate_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> list[int]:
    t = float(np.clip(t, 0.0, 1.0))
    return [
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    ]


def _value_to_color(value: float | None, higher_is_worse: bool) -> list[int]:
    if value is None or np.isnan(value):
        return [*APPLE_COLORS["gray"]]
    value = float(np.clip(value, 0.0, 1.0))
    if not higher_is_worse:
        value = 1.0 - value
    if value <= 0.5:
        return _interpolate_color(APPLE_COLORS["green"], APPLE_COLORS["yellow"], value / 0.5)
    return _interpolate_color(APPLE_COLORS["yellow"], APPLE_COLORS["red"], (value - 0.5) / 0.5)


def _compute_percentile_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return numeric.rank(pct=True)


def render_map(
    geo_df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    higher_is_worse: bool,
    highlight_lgas: Iterable[str] | None = None,
    view_state: pdk.ViewState | None = None,
    **_: object,
) -> None:
    """Render choropleth map with metric-driven colors and plain-language tooltip."""

    styled = geo_df.copy()
    if metric_key not in styled.columns:
        # Fallback: fill with NaN so we render neutral gray instead of empty map
        styled[metric_key] = np.nan

    styled["metric_percentile"] = _compute_percentile_series(styled[metric_key])
    styled["fill_color"] = styled["metric_percentile"].apply(
        lambda value: _value_to_color(value, higher_is_worse) + [255],
    )
    highlight_set = set(highlight_lgas or [])
    styled["line_color"] = styled["lga_name"].apply(
        lambda name: [*APPLE_COLORS["highlight"], 255] if name in highlight_set else [*APPLE_COLORS["ink"], 230],
    )
    styled["line_width"] = styled["lga_name"].apply(lambda name: 3.5 if name in highlight_set else 2.0)

    def _display_value(value: float | None, digits: int = 2) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "Not available"
        return f"{float(value):.{digits}f}"

    styled["metric_display"] = styled[metric_key].apply(_display_value)
    styled["u5mr_display"] = styled["u5mr_mean"].apply(lambda val: _display_value(val, 1))
    styled["facilities_display"] = styled["facilities_per_10k"].apply(_display_value)
    styled["towers_display"] = styled["towers_per_10k"].apply(_display_value)
    # Use GeoInterface dict (safer than JSON string for pydeck property access)
    geojson = styled.__geo_interface__
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width="properties.line_width",
        lineWidthUnits="pixels",
        lineWidthScale=1,
        lineWidthMinPixels=1.5,
    )
    if view_state is None:
        view_state = pdk.ViewState(latitude=9.0, longitude=8.7, zoom=5)
    elif isinstance(view_state, dict):
        view_state = pdk.ViewState(**view_state)
    tooltip = {
        "html": (
            "<b>{lga_name}</b>"
            "<br/>State: {state_name}"
            f"<br/>{metric_label}: {{metric_display}}"
            "<br/>Under-5 mortality: {u5mr_display}"
            "<br/>Facilities per 10k: {facilities_display}"
            "<br/>Connectivity towers per 10k: {towers_display}"
        ),
    }
    with st.spinner("Loading map..."):
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style=MAP_STYLE,
            )
        )


def render_summary_stats(features_df: pd.DataFrame) -> None:
    st.metric("LGAs", len(features_df))
    u5mr_mean = pd.to_numeric(features_df.get("u5mr_mean"), errors="coerce").mean()
    facilities_mean = pd.to_numeric(features_df.get("facilities_per_10k"), errors="coerce").mean()
    st.metric("Avg U5MR", f"{u5mr_mean:.1f}" if not np.isnan(u5mr_mean) else "Not available")
    st.metric(
        "Avg Facilities/10k",
        f"{facilities_mean:.2f}" if not np.isnan(facilities_mean) else "Not available",
    )


def render_shap_detail(shap_df: pd.DataFrame | None, lga_name: str, year: int | None = None) -> None:
    if shap_df is None:
        st.info("SHAP not available.")
        return
    row = shap_df[shap_df["lga_name"] == lga_name]
    if year is not None and "year" in row.columns:
        row = row[row["year"] == year]
    if row.empty:
        st.info("No SHAP data for this LGA.")
        return
    st.subheader("Top drivers (SHAP)")
    values = row.drop(columns=["lga_name"] + (["year"] if "year" in row.columns else [])).T
    values = values.rename(columns={row.index[0]: "shap_value"})
    values["abs"] = values["shap_value"].abs()
    top = values.sort_values("abs", ascending=False).head(8).reset_index(names="feature")
    st.bar_chart(top.set_index("feature")["shap_value"])
