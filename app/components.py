"""Streamlit UI components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

APPLE_COLORS = {
    "green": (52, 199, 89),
    "yellow": (255, 204, 0),
    "red": (255, 59, 48),
}


def _interpolate_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> list[int]:
    t = float(np.clip(t, 0.0, 1.0))
    return [
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    ]


def risk_to_color(value: float) -> list[int]:
    if np.isnan(value):
        return [200, 200, 200]
    value = float(np.clip(value, 0.0, 1.0))
    if value <= 0.5:
        return _interpolate_color(APPLE_COLORS["green"], APPLE_COLORS["yellow"], value / 0.5)
    return _interpolate_color(APPLE_COLORS["yellow"], APPLE_COLORS["red"], (value - 0.5) / 0.5)


def render_map(geo_df: pd.DataFrame, color_col: str = "risk_prob") -> None:
    """Render choropleth map of LGA risk."""

    styled = geo_df.copy()
    styled["fill_color"] = (
        styled[color_col]
        .fillna(np.nan)
        .apply(lambda value: risk_to_color(value) + [160])
    )
    geojson = styled.to_json()
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        auto_highlight=True,
        get_fill_color="properties.fill_color",
        get_line_color=[50, 50, 50],
        line_width_min_pixels=0.8,
    )
    view_state = pdk.ViewState(latitude=9.0, longitude=8.7, zoom=5)
    tooltip = {
        "html": (
            "<b>{lga_name}</b>" "<br/>State: {state_name}" "<br/>U5MR: {u5mr_mean}"
            "<br/>Risk: {risk_prob}"
            "<br/>Facilities/10k: {facilities_per_10k}"
            "<br/>Towers/10k: {towers_per_10k}"
            "<br/>Pop density: {population_density}"
        )
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


def render_summary_stats(features_df: pd.DataFrame) -> None:
    st.metric("LGAs", len(features_df))
    st.metric("Avg U5MR", f"{features_df['u5mr_mean'].mean():.1f}")
    st.metric("Avg Facilities/10k", f"{features_df['facilities_per_10k'].mean():.2f}")


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
