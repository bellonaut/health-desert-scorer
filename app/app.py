"""Streamlit app for Nigeria Health Desert Risk Scorer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import geopandas as gpd
import pandas as pd
import streamlit as st

from components import render_map, render_shap_detail, render_summary_stats


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
    if "year" not in merged.columns:
        merged["year"] = 2018
    if "state_name" not in merged.columns:
        merged["state_name"] = "Unknown"
    return merged, features, shap_df


def _get_version_label() -> str:
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        result = "unknown"
    return f"v{result}"


def _ensure_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_prob"] = pd.to_numeric(df.get("risk_prob", np.nan), errors="coerce")
    df["u5mr_mean"] = pd.to_numeric(df.get("u5mr_mean", np.nan), errors="coerce")
    df["facilities_per_10k"] = pd.to_numeric(df.get("facilities_per_10k", np.nan), errors="coerce")
    df["towers_per_10k"] = pd.to_numeric(df.get("towers_per_10k", np.nan), errors="coerce")
    df["population_density"] = pd.to_numeric(df.get("population_density", np.nan), errors="coerce")
    return df


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    st.set_page_config(page_title="Nigeria Health Desert Risk Scorer", layout="wide")
    st.title("Nigeria Health Desert Risk Scorer")

    st.markdown(
        """
        <style>
        :root { --surface: #FFFFFF; --background: #F5F5F7; --text: #1D1D1F; --text-muted: #86868B; }
        .app-card {
            background: var(--surface);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 8px rgba(0,0,0,0.06);
        }
        .app-muted { color: var(--text-muted); font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    merged, features, shap_df = load_data()
    merged = _ensure_display_columns(merged)

    years = sorted(merged["year"].dropna().unique().tolist())
    year = st.radio("Year", years, horizontal=True)

    left, right = st.columns([2.6, 1.4])
    with right:
        st.subheader("Filters")
        states = ["All"] + sorted(merged["state_name"].dropna().unique())
        state = st.selectbox("State", states)
        search = st.text_input("Search LGA")
        st.caption("Filter LGAs by state or name, then explore details.")

    filtered = merged[merged["year"] == year]
    if state != "All":
        filtered = filtered[filtered["state_name"] == state]
    if search:
        filtered = filtered[filtered["lga_name"].str.contains(search, case=False, na=False)]

    with left:
        st.subheader("Map")
        render_map(filtered)
        st.markdown("<div class='app-muted'>Hover over an LGA for details.</div>", unsafe_allow_html=True)

    with right:
        st.subheader("Summary")
        render_summary_stats(filtered)

        st.subheader("Compare LGAs")
        display_cols = [
            "lga_name",
            "state_name",
            "u5mr_mean",
            "risk_prob",
            "facilities_per_10k",
            "towers_per_10k",
            "population_density",
        ]
        editable = filtered[display_cols].copy()
        editable.insert(0, "select", False)
        edited = st.data_editor(editable, use_container_width=True, hide_index=True)
        selected_rows = edited[edited["select"]].head(4)
        if len(selected_rows) > 4:
            st.warning("Comparison is limited to 4 LGAs.")
        if not selected_rows.empty:
            st.markdown("#### Comparison cards")
            for _, row in selected_rows.iterrows():
                st.markdown(
                    f"""
                    <div class="app-card">
                        <strong>{row['lga_name']}</strong> · {row['state_name']}
                        <br/>U5MR: {_fmt(row['u5mr_mean'], 1)}
                        <br/>Risk: {_fmt(row['risk_prob'], 2)}
                        <br/>Facilities/10k: {_fmt(row['facilities_per_10k'], 2)}
                        <br/>Towers/10k: {_fmt(row['towers_per_10k'], 2)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            csv = selected_rows.drop(columns=["select"]).to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "lga_comparison.csv", "text/csv")
        else:
            st.info("Select LGAs in the table to compare.")

    st.header("LGA Detail")
    lga_options = sorted(filtered["lga_name"].dropna().unique())
    if lga_options:
        selected = st.selectbox("Select LGA", lga_options)
        detail = filtered[filtered["lga_name"] == selected].iloc[0]
        st.write(
            {
                "u5mr_mean": detail["u5mr_mean"],
                "facilities_per_10k": detail["facilities_per_10k"],
                "avg_distance_km": detail.get("avg_distance_km"),
                "risk_prob": detail["risk_prob"],
            }
        )
        render_shap_detail(shap_df, selected, year=year if isinstance(year, int) else None)
    else:
        st.info("No LGAs match the current filters.")

    st.markdown(
        f"<div class='app-muted'>Model version {_get_version_label()} · Data year {year}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
