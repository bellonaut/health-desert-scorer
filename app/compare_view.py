"""Comparison table view for Health Desert app."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


def render_compare_table(filtered_df: pd.DataFrame, selected_uids: Iterable[str], depth_level: int) -> None:
    uid_set = {str(uid) for uid in selected_uids}
    subset = filtered_df[filtered_df["lga_uid"].astype(str).isin(uid_set)]

    st.subheader("Compare LGAs")
    if subset.empty:
        st.info("Select two or more LGAs to compare.")
        return

    columns = [
        "lga_name",
        "state_name",
        "risk_score",
        "u5mr_mean",
        "facilities_per_10k",
        "avg_distance_km",
        "towers_per_10k",
        "coverage_5km",
    ]
    columns = [col for col in columns if col in subset.columns]
    display = subset[columns].copy()
    display = display.rename(
        columns={
            "risk_score": "Risk score",
            "u5mr_mean": "U5MR",
            "facilities_per_10k": "Facilities/10k",
            "avg_distance_km": "Avg distance (km)",
            "towers_per_10k": "Towers/10k",
            "coverage_5km": "5km coverage",
        }
    )
    st.dataframe(display.set_index("lga_name"))

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download comparison CSV",
        data=csv_bytes,
        file_name="compare_lgas.csv",
        mime="text/csv",
        key="hd_compare_download",
    )
