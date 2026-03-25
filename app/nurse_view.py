"""Hotspot list, detail drawer, and action prompts."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

FOCUS_COLUMN = {
    "All risk": "risk_score",
    "Child mortality": "u5mr_mean",
    "Facility access": "facilities_per_10k",
    "Connectivity": "towers_per_10k",
    "5km coverage": "coverage_5km",
}

ASCENDING_FOCUS = {"facilities_per_10k", "towers_per_10k", "coverage_5km"}

HIGH_RISK = 0.7
HIGH_MORTALITY = 70.0
LOW_FACILITIES = 1.0
HIGH_DISTANCE = 5.0
LOW_TOWERS = 0.5
LOW_COVERAGE = 250.0
COLOR_GOOD = "#0072B2"
COLOR_WARN = "#E69F00"
COLOR_BAD = "#D55E00"


def render_focus_filter() -> str:
    return st.radio(
        "Focus",
        list(FOCUS_COLUMN.keys()),
        horizontal=True,
        key="hd_focus",
        label_visibility="collapsed",
    )


def get_ranked_lgas(df: pd.DataFrame, focus: str) -> pd.DataFrame:
    column = FOCUS_COLUMN.get(focus, "risk_score")
    ascending = column in ASCENDING_FOCUS
    ranked = df.sort_values(column, ascending=ascending, na_position="last")
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def _risk_badge(score: float | None) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return '<span style="padding:2px 8px;border-radius:4px;font-size:11px;font-family:monospace;background:#1f2937;color:#e5e7eb;">NA</span>'
    color = COLOR_BAD if score > 0.66 else COLOR_WARN if score > 0.33 else COLOR_GOOD
    return (
        f'<span style="background:{color}20;color:{color};padding:2px 8px;'
        f'border-radius:4px;font-size:11px;font-family:monospace">{score*100:.0f}</span>'
    )


def _model_version_label(model_version: str | list[str] | tuple[str, ...] | None) -> str:
    if isinstance(model_version, (list, tuple)):
        cleaned = [str(item) for item in model_version if item]
        return ", ".join(cleaned) if cleaned else "Unknown"
    if model_version:
        return str(model_version)
    return "Unknown"


def render_methodology_panel(
    features_df: pd.DataFrame,
    active_year: str | int | None = None,
    model_version: str | list[str] | tuple[str, ...] | None = None,
) -> None:
    scoped = features_df.copy()
    if "lga_uid" in scoped.columns:
        scoped = scoped.drop_duplicates(subset=["lga_uid"])

    if active_year is None and "year" in scoped.columns:
        years = [str(year) for year in pd.to_numeric(scoped["year"], errors="coerce").dropna().astype(int).unique().tolist()]
        active_year = years[0] if len(years) == 1 else "Multiple"

    if model_version is None:
        attr_version = features_df.attrs.get("model_version") if hasattr(features_df, "attrs") else None
        if attr_version:
            model_version = attr_version
        elif "model_version" in scoped.columns:
            versions = scoped["model_version"].dropna().astype(str).unique().tolist()
            model_version = versions or None

    risk_series = pd.to_numeric(scoped.get("risk_score"), errors="coerce")
    lga_count = int(scoped["lga_uid"].nunique()) if "lga_uid" in scoped.columns else int(len(scoped))
    median_risk = float(risk_series.median()) if risk_series.notna().any() else None

    with st.container(border=True):
        st.caption("METHODOLOGY")
        stat1, stat2 = st.columns(2)
        stat3, stat4 = st.columns(2)
        with stat1:
            st.metric("Model version", _model_version_label(model_version))
        with stat2:
            st.metric("Active year", str(active_year or "Unknown"))
        with stat3:
            st.metric("LGAs scored", f"{lga_count}")
        with stat4:
            st.metric("Median risk", f"{median_risk:.3f}" if median_risk is not None else "NA")

        st.markdown(_risk_badge(median_risk), unsafe_allow_html=True)
        st.markdown(
            (
                "The composite score combines facilities per 10,000 people, routed travel time access, "
                "child mortality, connectivity, and 5km coverage into a single access-risk signal. "
                "Each input is normalized onto a common scale and then weighted into a 0-1 score, "
                "where higher values indicate greater barriers to reaching care. "
                "It is a planning aid, not a diagnosis, and should be used alongside local operational knowledge."
            )
        )


def render_download_button(top_lgas: pd.DataFrame, focus: str) -> None:
    export_cols = ["rank", "lga_name", "state_name", "risk_score", FOCUS_COLUMN.get(focus, "risk_score")]
    export_cols = [col for col in export_cols if col in top_lgas.columns]
    csv_bytes = top_lgas.head(10)[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranked list (CSV)",
        data=csv_bytes,
        file_name="health_desert_hotspots.csv",
        mime="text/csv",
        key="hd_rank_download",
    )


def _percentile(series: pd.Series, value: float | None) -> Optional[int]:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    pct = (numeric <= float(value)).sum() / len(numeric) * 100
    return int(round(pct))


def percentile_bar(label: str, value: float | None, series: pd.Series, higher_is_better: bool = True) -> None:
    pct = _percentile(series, value)
    if pct is None:
        pct = 0
        display = "—"
    else:
        display_pct = pct if higher_is_better else (100 - pct)
        pct = int(np.clip(display_pct, 0, 100))
        display = f"{pct}%"
    col_label, col_bar, col_val = st.columns([3, 5, 1])
    with col_label:
        st.caption(label)
    with col_bar:
        st.progress(pct / 100)
    with col_val:
        st.caption(display)


def format_value(value: float | int | None, suffix: str = "", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _safe_delta(value: float | int | None, median: float | int | None) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(median, (int, float)):
        return None
    if np.isnan(value) or np.isnan(median):
        return None
    return float(value) - float(median)


def generate_action_prompt(lga: pd.Series) -> str:
    prompts: list[str] = []
    risk = lga.get("risk_score")
    if isinstance(risk, (int, float)) and not np.isnan(risk) and risk >= HIGH_RISK:
        prompts.append("Flag for rapid assessment; combined risk is very high.")
    mortality = lga.get("u5mr_mean")
    if isinstance(mortality, (int, float)) and not np.isnan(mortality) and mortality >= HIGH_MORTALITY:
        prompts.append("Prioritise child health outreach; mortality exceeds the alert threshold.")
    facilities = lga.get("facilities_per_10k")
    if isinstance(facilities, (int, float)) and not np.isnan(facilities) and facilities <= LOW_FACILITIES:
        prompts.append("Consider temporary clinics or referrals; facility density is low.")
    distance = lga.get("avg_distance_km")
    if isinstance(distance, (int, float)) and not np.isnan(distance) and distance >= HIGH_DISTANCE:
        prompts.append("Improve transport or site new facilities; travel distance is long.")
    towers = lga.get("towers_per_10k")
    if isinstance(towers, (int, float)) and not np.isnan(towers) and towers <= LOW_TOWERS:
        prompts.append("Use radio/SMS strategies; connectivity is weak.")
    coverage = lga.get("coverage_5km")
    if isinstance(coverage, (int, float)) and not np.isnan(coverage) and coverage <= LOW_COVERAGE:
        prompts.append("Map catchment gaps; 5km facility coverage is low.")
    if not prompts:
        return "Maintain routine monitoring and coordinate with local health teams for next review."
    return " ".join(prompts)


def render_detail_panel(
    lga: pd.Series,
    features: pd.DataFrame,
    depth_level: int,
    shap_values: Optional[dict[str, float]] = None,
) -> None:
    with st.expander(f"📍 {lga['lga_name']}, {lga['state_name']}", expanded=True):
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
    medians = features[
        ["facilities_per_10k", "avg_distance_km", "towers_per_10k", "u5mr_mean", "risk_score"]
    ].median(numeric_only=True)

    with m1:
        delta = _safe_delta(lga.get("facilities_per_10k"), medians.get("facilities_per_10k"))
        st.metric(
            "Facilities / 10k",
            format_value(lga.get("facilities_per_10k")),
            delta=f"{delta:+.1f}" if isinstance(delta, (int, float)) else None,
            delta_color="normal",
        )
    with m2:
        delta = _safe_delta(lga.get("avg_distance_km"), medians.get("avg_distance_km"))
        st.metric(
            "Avg distance",
            format_value(lga.get("avg_distance_km"), suffix=" km"),
            delta=f"{delta:+.1f} km" if isinstance(delta, (int, float)) else None,
            delta_color="inverse",
        )
    with m3:
        delta = _safe_delta(lga.get("towers_per_10k"), medians.get("towers_per_10k"))
        st.metric(
            "Towers / 10k",
            format_value(lga.get("towers_per_10k")),
            delta=f"{delta:+.2f}" if isinstance(delta, (int, float)) else None,
            delta_color="normal",
        )
    with m4:
        delta = _safe_delta(lga.get("risk_score"), medians.get("risk_score"))
        st.metric(
            "Risk score",
            format_value(lga.get("risk_score"), digits=2),
            delta=f"{delta:+.2f}" if isinstance(delta, (int, float)) else None,
            delta_color="inverse",
        )

        st.divider()
        percentile_bar("Under-5 mortality", lga.get("u5mr_mean"), features.get("u5mr_mean"), higher_is_better=False)
        percentile_bar("Facilities / 10k", lga.get("facilities_per_10k"), features.get("facilities_per_10k"), higher_is_better=True)
        percentile_bar("Travel distance", lga.get("avg_distance_km"), features.get("avg_distance_km"), higher_is_better=False)
        percentile_bar("Connectivity", lga.get("towers_per_10k"), features.get("towers_per_10k"), higher_is_better=True)
        percentile_bar("5km coverage", lga.get("coverage_5km"), features.get("coverage_5km"), higher_is_better=True)

        st.info(generate_action_prompt(lga))
        st.caption("Decision-support only. Always combine with local knowledge and community input.")

        if depth_level >= 2:
            st.divider()
            st.caption("FEATURE CONTRIBUTION (SHAP)")
            if shap_values:
                for feature, value in shap_values.items():
                    col_feat, col_bar = st.columns([3, 5])
                    with col_feat:
                        st.caption(feature)
                    with col_bar:
                        st.progress(min(abs(value) * 2, 1.0))
            else:
                st.caption("SHAP details not available for this LGA.")


def render_compare_strip(option_labels: dict[str, str]) -> tuple[list[str], bool]:
    st.divider()
    st.caption("COMPARE LGAS")
    selection = st.multiselect(
        "",
        list(option_labels.keys()),
        format_func=lambda uid: option_labels.get(uid, uid),
        max_selections=4,
        key="hd_compare_lgas",
        label_visibility="collapsed",
    )
    trigger = False
    if len(selection) >= 2:
        if st.button("RUN COMPARISON →", key="hd_compare_go"):
            trigger = True
    return selection, trigger
