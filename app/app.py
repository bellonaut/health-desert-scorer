"""Streamlit app for Nigeria Health Desert Risk Scorer."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from app.components import render_map, render_shap_detail, render_summary_stats


@st.cache_data
def load_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame | None]:
    lga_path = Path("data/raw/lga_boundaries.geojson")
    features_path = Path("data/processed/lga_features.csv")
    preds_path = Path("data/processed/lga_predictions.csv")
    shap_path = Path("data/processed/shap_values.csv")

    lgas = gpd.read_file(lga_path)
    features = pd.read_csv(features_path)
    preds = pd.read_csv(preds_path)
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None

    merged = lgas.merge(preds, on="lga_name", how="left").merge(features, on="lga_name", how="left")
    return merged, features, shap_df


def main() -> None:
    st.set_page_config(page_title="Nigeria Health Desert Risk Scorer", layout="wide")
    st.title("Nigeria Health Desert Risk Scorer")

    merged, features, shap_df = load_data()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_map(merged)
    with col2:
        render_summary_stats(features)

    st.header("LGA Detail")
    selected = st.selectbox("Select LGA", sorted(merged["lga_name"].dropna().unique()))
    detail = merged[merged["lga_name"] == selected].iloc[0]
    st.write(
        {
            "u5mr_mean": detail["u5mr_mean"],
            "facilities_per_10k": detail["facilities_per_10k"],
            "avg_distance_km": detail["avg_distance_km"],
            "risk_prob": detail["risk_prob"],
        }
    )
    render_shap_detail(shap_df, selected)


if __name__ == "__main__":
    main()
