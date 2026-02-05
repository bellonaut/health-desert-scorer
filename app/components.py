"""Streamlit UI components."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st


def render_map(geo_df: pd.DataFrame, color_col: str = "risk_prob") -> None:
    """Render choropleth map of LGA risk."""

    geojson = geo_df.to_json()
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        auto_highlight=True,
        get_fill_color=f"[255 * {color_col}, 50, 180, 160]",
        get_line_color=[50, 50, 50],
    )
    view_state = pdk.ViewState(latitude=9.0, longitude=8.7, zoom=5)
    tooltip = {"html": "<b>{lga_name}</b><br/>Risk: {risk_prob}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


def render_summary_stats(features_df: pd.DataFrame) -> None:
    st.metric("LGAs", len(features_df))
    st.metric("Avg U5MR", f"{features_df['u5mr_mean'].mean():.1f}")
    st.metric("Avg Facilities/10k", f"{features_df['facilities_per_10k'].mean():.2f}")


def render_shap_detail(shap_df: pd.DataFrame | None, lga_name: str) -> None:
    if shap_df is None:
        st.info("SHAP not available.")
        return
    row = shap_df[shap_df["lga_name"] == lga_name]
    if row.empty:
        st.info("No SHAP data for this LGA.")
        return
    st.subheader("Top drivers (SHAP)")
    values = row.drop(columns=["lga_name"]).T.rename(columns={row.index[0]: "shap_value"})
    values["abs"] = values["shap_value"].abs()
    st.dataframe(values.sort_values("abs", ascending=False).head(5))
