"""Streamlit app for Nigeria Health Desert Risk Scorer."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
import sys

import numpy as np

import geopandas as gpd
import pandas as pd
import streamlit as st

# Ensure app directory is importable when Streamlit launches from project root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components import render_map, render_shap_detail, render_summary_stats

MetricDefinitions = dict[str, dict[str, object]]


@st.cache_data
def load_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame | None]:
    lga_path = Path("data/raw/lga_boundaries.geojson")
    features_path = Path("data/processed/lga_features.csv")
    preds_path = Path("data/processed/lga_predictions.csv")
    shap_path = Path("data/processed/shap_values.csv")

    lgas = gpd.read_file(lga_path)
    # Normalize column names from common alternatives
    rename_map: dict[str, str] = {}
    if "lga_name" not in lgas.columns:
        for cand in ["lganame", "lga", "adm2_name", "name", "lg_name", "NAME_2"]:
            if cand in lgas.columns:
                rename_map[cand] = "lga_name"
                break
    if "state_name" not in lgas.columns:
        for cand in ["statename", "state", "adm1_name", "NAME_1"]:
            if cand in lgas.columns:
                rename_map[cand] = "state_name"
                break
    if rename_map:
        lgas = lgas.rename(columns=rename_map)
    if "lga_name" not in lgas.columns:
        raise KeyError("Could not find LGA name column in boundaries (tried lganame/lga/adm2/name).")
    lgas["lga_name"] = lgas["lga_name"].astype(str).str.strip()
    if "state_name" in lgas:
        lgas["state_name"] = lgas["state_name"].astype(str).str.strip()
    features = pd.read_csv(features_path)
    preds = pd.read_csv(preds_path)
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None

    # Merge predictions and features by both lga_name and year when possible to avoid duplicates
    preds_has_year = "year" in preds.columns
    feats_has_year = "year" in features.columns
    if preds_has_year and feats_has_year:
        merged = lgas.merge(preds, on="lga_name", how="left").merge(
            features, on=["lga_name", "year"], how="left"
        )
    else:
        merged = lgas.merge(preds, on="lga_name", how="left").merge(features, on="lga_name", how="left")

    # Coalesce any suffixed year columns
    if "year" not in merged.columns:
        merged["year"] = merged.get("year_x") if "year_x" in merged else merged.get("year_y", 2018)
    if "year_x" in merged.columns:
        merged = merged.drop(columns="year_x")
    if "year_y" in merged.columns:
        merged = merged.drop(columns="year_y")

    merged = merged.drop_duplicates(subset=["lga_name", "year"])

    return merged, features, shap_df


def get_metric_definitions() -> MetricDefinitions:
    return {
        "risk_prob": {
            "label": "Combined risk score",
            "description": "Model probability that an area is underserved; higher means higher risk.",
            "unit": "probability",
            "higher_is_worse": True,
        },
        "u5mr_mean": {
            "label": "Under-5 mortality",
            "description": "Estimated deaths of children under five per 1,000 live births in this area.",
            "unit": "deaths per 1,000 births",
            "higher_is_worse": True,
        },
        "facilities_per_10k": {
            "label": "Facilities per 10k people",
            "description": "Number of health facilities per 10,000 residents; higher means better access.",
            "unit": "facilities per 10k",
            "higher_is_worse": False,
        },
        "avg_distance_km": {
            "label": "Average distance to care",
            "description": "Average distance in kilometers to the nearest health facility; lower is better.",
            "unit": "km",
            "higher_is_worse": True,
        },
        "towers_per_10k": {
            "label": "Connectivity towers per 10k",
            "description": "Cell towers per 10,000 residents, used as a proxy for digital connectivity.",
            "unit": "towers per 10k",
            "higher_is_worse": False,
        },
        "access_score": {
            "label": "Access to care score",
            "description": "Combined signal from facilities and travel distance; higher means weaker access.",
            "unit": "score",
            "higher_is_worse": True,
        },
        "population_density": {
            "label": "Population density",
            "description": "People per square kilometer, useful for context but not a risk signal by itself.",
            "unit": "people per km²",
            "higher_is_worse": False,
        },
    }


def format_metric(value: float | int | None, unit: str, fallback: str = "Not available") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    if unit == "probability":
        return f"{float(value):.2f}"
    if unit in {"km", "facilities per 10k", "towers per 10k", "people per km²"}:
        return f"{float(value):.2f}"
    return f"{float(value):.1f}"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "towers_per_10k" not in df.columns:
        if "towers_per_10k_pop" in df.columns:
            df["towers_per_10k"] = df["towers_per_10k_pop"]
        else:
            tower_count_col = next(
                (col for col in ["towers_count", "tower_count"] if col in df.columns),
                None,
            )
            pop_col = next(
                (col for col in ["population", "population_total", "pop_total"] if col in df.columns),
                None,
            )
            if tower_count_col and pop_col:
                df["towers_per_10k"] = pd.to_numeric(df[tower_count_col], errors="coerce") / pd.to_numeric(
                    df[pop_col],
                    errors="coerce",
                ) * 10000

    if "facilities_per_10k" not in df.columns:
        facility_count_col = next(
            (col for col in ["facilities_count", "facility_count"] if col in df.columns),
            None,
        )
        pop_col = next(
            (col for col in ["population", "population_total", "pop_total"] if col in df.columns),
            None,
        )
        if facility_count_col and pop_col:
            df["facilities_per_10k"] = pd.to_numeric(df[facility_count_col], errors="coerce") / pd.to_numeric(
                df[pop_col],
                errors="coerce",
            ) * 10000

    if "avg_distance_km" not in df.columns:
        for candidate in ["avg_distance", "avg_dist_km", "avg_distance_to_facility_km"]:
            if candidate in df.columns:
                df["avg_distance_km"] = df[candidate]
                break

    if "state_name" not in df.columns:
        for candidate in ["state", "state_name_x", "state_name_y", "state_name_feat", "state_name_lga"]:
            if candidate in df.columns:
                df["state_name"] = df[candidate]
                break
    if "state_name" not in df.columns:
        raise KeyError("State name is required but missing from the merged dataset.")

    numeric_fields = [
        "risk_prob",
        "u5mr_mean",
        "facilities_per_10k",
        "towers_per_10k",
        "population_density",
        "avg_distance_km",
    ]
    for field in numeric_fields:
        df[field] = pd.to_numeric(df.get(field, np.nan), errors="coerce")

    df["state_name"] = df["state_name"].astype(str).str.strip()
    return df


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    metric_defs = get_metric_definitions()
    for metric in metric_defs:
        if metric in df.columns:
            df[f"{metric}_pct"] = pd.to_numeric(df[metric], errors="coerce").rank(pct=True)
    return df


def _get_version_label() -> str:
    try:
        result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        result = "unknown"
    return f"v{result}"


def _build_access_score(df: pd.DataFrame) -> pd.Series:
    facilities_pct = pd.to_numeric(df.get("facilities_per_10k"), errors="coerce").rank(pct=True)
    distance_pct = pd.to_numeric(df.get("avg_distance_km"), errors="coerce").rank(pct=True)
    if distance_pct.notna().sum() == 0:
        return 1 - facilities_pct
    return (1 - facilities_pct + distance_pct) / 2


def _metric_badge(value_pct: float | None, higher_is_worse: bool) -> str:
    if value_pct is None or np.isnan(value_pct):
        return "Not available"
    adjusted = value_pct if higher_is_worse else 1 - value_pct
    if adjusted <= 0.33:
        return "Low"
    if adjusted <= 0.66:
        return "Medium"
    return "High"


def _takeaway(row: pd.Series) -> str:
    u5 = row.get("u5mr_mean_pct")
    towers = row.get("towers_per_10k_pct")
    facilities = row.get("facilities_per_10k_pct")
    access_score = row.get("access_score_pct")
    statements = []
    if u5 is not None and u5 >= 0.66:
        statements.append("High mortality")
    if towers is not None and towers <= 0.33:
        statements.append("Low connectivity")
    if access_score is not None and access_score >= 0.66:
        statements.append("Weak access to care")
    elif facilities is not None and facilities <= 0.33:
        statements.append("Few facilities")
    if not statements:
        return "Mixed signals across mortality, access, and connectivity."
    return f"{' + '.join(statements)}: likely underserved."


def _progress_value(value_pct: float | None, higher_is_worse: bool) -> float:
    if value_pct is None or np.isnan(value_pct):
        return 0.0
    value = float(value_pct)
    if not higher_is_worse:
        value = 1 - value
    return float(np.clip(value, 0.0, 1.0))


def _render_color_legend() -> None:
    st.markdown(
        """
        <div style="margin-top:0.5rem;margin-bottom:0.5rem;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:0.85rem;color:#4b4b4b;">Better</span>
                <div style="flex:1;height:10px;border-radius:6px;background:linear-gradient(90deg,#3DAB74,#F7C548,#E15E54);"></div>
                <span style="font-size:0.85rem;color:#4b4b4b;">Worse</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_view_state(filtered: gpd.GeoDataFrame, selected_lgas: list[str]) -> object | None:
    if not selected_lgas:
        return None
    subset = filtered[filtered["lga_name"].isin(selected_lgas)]
    if subset.empty or "geometry" not in subset:
        return None
    bounds = subset.total_bounds
    if bounds is None or len(bounds) != 4:
        return None
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    return {"latitude": center_lat, "longitude": center_lon, "zoom": 6.2}


def main() -> None:
    st.set_page_config(page_title="Nigeria Health Desert Risk Scorer", layout="wide")
    st.markdown(
        """
        <style>
        :root { --surface: #FFFFFF; --background: #F6F7F9; --text: #1D1D1F; --text-muted: #6E7177; --accent: #008C8F; }
        .block-container { padding-top: 2.4rem; }
        .app-card {
            background: var(--surface);
            border-radius: 14px;
            padding: 16px;
            border: 1px solid rgba(0,0,0,0.04);
            box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.05);
        }
        .app-muted { color: var(--text-muted); font-size: 0.85rem; }
        .app-hero {
            background: linear-gradient(120deg, rgba(0,140,143,0.08), rgba(255,255,255,0.9));
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(0,0,0,0.04);
        }
        .app-hero h1 { margin-bottom: 0.2rem; font-size: 2rem; }
        .app-hero span { color: var(--accent); font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-hero">
            <h1>Nigeria Health Desert Risk Scorer</h1>
            <div class="app-muted">
                A research-grade view of where <span>child mortality</span>, <span>healthcare access</span>, and
                <span>digital connectivity</span> overlap.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "onboarding_seen" not in st.session_state:
        st.session_state["onboarding_seen"] = False
    if "onboarding_toast_shown" not in st.session_state:
        st.session_state["onboarding_toast_shown"] = False
    if not st.session_state["onboarding_seen"] and not st.session_state["onboarding_toast_shown"]:
        st.toast("Welcome. This tool helps explore where child mortality, healthcare access, and connectivity intersect.")
        st.session_state["onboarding_toast_shown"] = True
    if not st.session_state["onboarding_seen"]:
        with st.expander("Welcome — start here", expanded=True):
            st.info(
                "This tool helps explore where child mortality, healthcare access, and digital connectivity intersect "
                "across Nigeria."
            )
            st.markdown("**Step 1 → Choose what you want to explore**")
            st.markdown("**Step 2 → Click a hotspot suggestion**")
            st.markdown("**Step 3 → Compare places side by side**")
            if st.button("Got it — explore"):
                st.session_state["onboarding_seen"] = True

    merged, features, shap_df = load_data()
    merged = standardize_columns(merged)
    merged["access_score"] = _build_access_score(merged)
    merged = compute_percentiles(merged)
    metric_defs = get_metric_definitions()

    years = sorted(merged["year"].dropna().unique().tolist())
    year = st.radio("Year", years, horizontal=True)
    filtered = merged[merged["year"] == year]

    guided_tab, explore_tab = st.tabs(["Guided Tour (Recommended)", "Explore Freely"])

    with guided_tab:
        st.subheader("Guided Tour: spot patterns in 60 seconds")
        st.caption("Follow the three steps below to see where mortality, access, and connectivity overlap.")

        storyline_cols = st.columns(3)
        storyline_examples = [
            {
                "title": "Highest combined risk",
                "description": "Areas where the model flags the strongest risk signals.",
                "lga_list": filtered.sort_values("risk_prob", ascending=False)["lga_name"].head(3).tolist(),
            },
            {
                "title": "Lowest connectivity",
                "description": "Places with the fewest towers per 10k people.",
                "lga_list": filtered.sort_values("towers_per_10k", ascending=True)["lga_name"].head(3).tolist(),
            },
            {
                "title": "Weakest access to care",
                "description": "Low facilities + long travel distances.",
                "lga_list": filtered.sort_values("access_score", ascending=False)["lga_name"].head(3).tolist(),
            },
        ]
        for col, story in zip(storyline_cols, storyline_examples, strict=False):
            with col:
                st.markdown(
                    f"""
                    <div class="app-card">
                        <strong>{story['title']}</strong>
                        <div class="app-muted">{story['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Load: {story['title']}", key=f"story-{story['title']}"):
                    with st.spinner("Finding focus areas…"):
                        time.sleep(0.2)
                    st.session_state["guided_selected_lgas"] = story["lga_list"]
                    st.session_state["guided_last_action"] = story["title"]

        st.divider()
        st.markdown("### Step 1: Choose what you want to see")
        guided_metric_choice = st.selectbox(
            "Pick a focus",
            [
                "Mortality (Under-5 mortality)",
                "Access to care (facilities + travel distance)",
                "Digital connectivity (towers per 10k)",
                "Combined risk score (model output)",
            ],
        )
        guided_metric_map = {
            "Mortality (Under-5 mortality)": "u5mr_mean",
            "Access to care (facilities + travel distance)": "access_score",
            "Digital connectivity (towers per 10k)": "towers_per_10k",
            "Combined risk score (model output)": "risk_prob",
        }
        guided_metric_key = guided_metric_map[guided_metric_choice]

        st.markdown("### Step 2: Find a hotspot")
        hotspot_cols = st.columns(4)
        with hotspot_cols[0]:
            if st.button("Show me the highest-risk areas"):
                with st.spinner("Finding areas with highest risk…"):
                    time.sleep(0.2)
                st.session_state["guided_selected_lgas"] = (
                    filtered.sort_values("risk_prob", ascending=False)["lga_name"].head(4).tolist()
                )
                st.session_state["guided_last_action"] = "Highest-risk areas"
        with hotspot_cols[1]:
            if st.button("Show me low-connectivity areas"):
                with st.spinner("Finding connectivity gaps…"):
                    time.sleep(0.2)
                st.session_state["guided_selected_lgas"] = (
                    filtered.sort_values("towers_per_10k", ascending=True)["lga_name"].head(4).tolist()
                )
                st.session_state["guided_last_action"] = "Low-connectivity areas"
        with hotspot_cols[2]:
            if st.button("Show me worst access to care"):
                with st.spinner("Finding access gaps…"):
                    time.sleep(0.2)
                st.session_state["guided_selected_lgas"] = (
                    filtered.sort_values("access_score", ascending=False)["lga_name"].head(4).tolist()
                )
                st.session_state["guided_last_action"] = "Weak access to care"
        with hotspot_cols[3]:
            if st.button("Surprise me (random example)"):
                if len(filtered) == 0:
                    st.warning("No LGAs available to sample.")
                else:
                    with st.spinner("Picking a surprise set…"):
                        time.sleep(0.2)
                    st.session_state["guided_selected_lgas"] = filtered.sample(min(3, len(filtered)))["lga_name"].tolist()
                    st.session_state["guided_last_action"] = "Surprise example"

        if st.session_state.get("guided_last_action"):
            st.success(f"{st.session_state['guided_last_action']} loaded. Compare them below.")

        selected_lgas = st.session_state.get("guided_selected_lgas", [])
        view_state = _get_view_state(filtered, selected_lgas)

        map_cols = st.columns([2.3, 1])
        with map_cols[0]:
            metric_info = metric_defs.get(guided_metric_key, {})
            render_map(
                filtered,
                metric_key=guided_metric_key,
                metric_label=str(metric_info.get("label", guided_metric_key)),
                higher_is_worse=bool(metric_info.get("higher_is_worse", True)),
                highlight_lgas=selected_lgas,
                view_state=view_state,
            )
            st.markdown(
                "<div class='app-muted'>Green = better outcomes · Red = worse outcomes. "
                "Outlined areas match your selection.</div>",
                unsafe_allow_html=True,
            )
        with map_cols[1]:
            st.markdown("**Legend**")
            _render_color_legend()
            st.markdown("- 🟩 Better / lower risk")
            st.markdown("- 🟨 Medium")
            st.markdown("- 🟥 Worse / higher risk")

        st.markdown("### Step 3: Compare places")
        compare_options = [
            f"{row.lga_name} ({row.state_name})"
            for row in filtered[["lga_name", "state_name"]].drop_duplicates().itertuples()
        ]
        default_compare = [
            option
            for option in compare_options
            if any(name in option for name in selected_lgas)
        ][:4]
        selected_compare = st.multiselect(
            "Pick 2–4 places to compare",
            compare_options,
            default=default_compare,
            max_selections=4,
        )
        compare_lgas = [option.split(" (")[0] for option in selected_compare]
        comparison = filtered[filtered["lga_name"].isin(compare_lgas)]
        if comparison.empty:
            st.info("Select two or more places to compare.")
        else:
            comparison_cols = st.columns(max(1, len(comparison)))
            for col, (_, row) in zip(comparison_cols, comparison.iterrows(), strict=False):
                with col:
                    st.markdown(
                        f"""
                        <div class="app-card">
                            <strong>{row['lga_name']}</strong>
                            <div class="app-muted">{row['state_name']}</div>
                            <br/>❤️ Mortality: {_metric_badge(row.get('u5mr_mean_pct'), True)} ({format_metric(row['u5mr_mean'], metric_defs['u5mr_mean']['unit'])})
                            <br/>🏥 Access: {_metric_badge(row.get('access_score_pct'), True)}
                            <br/>📡 Connectivity: {_metric_badge(row.get('towers_per_10k_pct'), False)} ({format_metric(row['towers_per_10k'], metric_defs['towers_per_10k']['unit'])})
                            <br/><em>{_takeaway(row)}</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Risk signal strength**")
                    st.progress(_progress_value(row.get("risk_prob_pct"), True))

        with st.expander("What am I looking at?"):
            st.markdown("**Under-5 mortality** — Estimated deaths of children under five per 1,000 live births.")
            st.markdown("**Facilities per 10k** — A quick sense of how many clinics are nearby.")
            st.markdown("**Towers per 10k** — A proxy for digital connectivity and information access.")
            st.markdown("**Risk score** — A combined signal of higher mortality and weaker access.")
        with st.expander("Glossary"):
            st.markdown("- **LGA**: Local Government Area, a local administrative zone.")
            st.markdown("- **Proxy**: A stand-in measure when direct data is unavailable.")
            st.markdown("- **Percentile**: How a place ranks compared with all others.")

    with explore_tab:
        st.subheader("Explore the data yourself")
        st.caption("Use the filters to highlight patterns without needing geographic knowledge.")

        left, right = st.columns([2.4, 1.2])
        with right:
            st.markdown("**Filters**")
            states = sorted(filtered["state_name"].dropna().unique())
            state_choice = st.selectbox("State (optional)", ["All states"] + states)
            search_options = ["All locations"] + [f"State: {state}" for state in states]
            search_options += [
                f"LGA: {row.lga_name} ({row.state_name})"
                for row in filtered[["lga_name", "state_name"]].drop_duplicates().itertuples()
            ]
            search_choice = st.selectbox("Search by state or LGA", search_options)
            metric_choice = st.selectbox(
                "Color the map by",
                [metric_defs[key]["label"] for key in metric_defs if key != "population_density"],
            )
            metric_key = next(
                key for key, value in metric_defs.items() if value["label"] == metric_choice
            )
            highlight_pct = st.slider("Highlight top X% for this metric", 5, 50, 15, step=5)

        explore_filtered = filtered.copy()
        if state_choice != "All states":
            explore_filtered = explore_filtered[explore_filtered["state_name"] == state_choice]
        if search_choice.startswith("State: "):
            explore_filtered = explore_filtered[explore_filtered["state_name"] == search_choice.replace("State: ", "")]
        if search_choice.startswith("LGA: "):
            lga_name = search_choice.split("LGA: ")[1].split(" (")[0]
            explore_filtered = explore_filtered[explore_filtered["lga_name"] == lga_name]

        metric_pct_col = f"{metric_key}_pct"
        highlight_threshold = 1 - highlight_pct / 100
        highlight_lgas = explore_filtered.loc[
            explore_filtered[metric_pct_col] >= highlight_threshold,
            "lga_name",
        ].tolist()
        metric_info = metric_defs[metric_key]

        with left:
            render_map(
                explore_filtered,
                metric_key=metric_key,
                metric_label=str(metric_info["label"]),
                higher_is_worse=bool(metric_info["higher_is_worse"]),
                highlight_lgas=highlight_lgas,
            )
            st.markdown(
                "<div class='app-muted'>Outlined areas are in the top highlight band.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Legend**")
            _render_color_legend()
            st.markdown("- 🟩 Better / lower risk")
            st.markdown("- 🟨 Medium")
            st.markdown("- 🟥 Worse / higher risk")

        with right:
            st.subheader("Summary")
            render_summary_stats(explore_filtered)

            st.subheader("LGA Profile")
            lga_options = sorted(explore_filtered["lga_name"].dropna().unique())
            if lga_options:
                selected = st.selectbox("Select an LGA", lga_options)
                detail = explore_filtered[explore_filtered["lga_name"] == selected].iloc[0]
                national_medians = explore_filtered[
                    ["u5mr_mean", "facilities_per_10k", "avg_distance_km", "towers_per_10k", "risk_prob"]
                ].median(numeric_only=True)
                profile_items = [
                    ("Under-5 mortality", "u5mr_mean", True),
                    ("Facilities per 10k", "facilities_per_10k", False),
                    ("Average distance to care", "avg_distance_km", True),
                    ("Connectivity towers per 10k", "towers_per_10k", False),
                    ("Combined risk score", "risk_prob", True),
                ]
                for label, key, higher_is_worse in profile_items:
                    info = metric_defs.get(key, {})
                    value = detail.get(key)
                    delta = None
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        median = national_medians.get(key)
                        if isinstance(median, (int, float)) and not np.isnan(median):
                            delta = value - median
                    st.markdown(
                        f"""
                        <div class="app-card">
                            <strong>{label}</strong>
                            <div>{format_metric(value, info.get("unit", ""))}</div>
                            <div class="app-muted">{info.get("description", "Metric description not available.")}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.metric(
                        f"{label} vs median",
                        format_metric(value, info.get("unit", "")),
                        delta=f"{delta:+.2f}" if isinstance(delta, (int, float)) else "n/a",
                        delta_color="inverse" if higher_is_worse else "normal",
                    )
                    st.progress(_progress_value(detail.get(f"{key}_pct"), higher_is_worse))
                render_shap_detail(shap_df, selected, year=year if isinstance(year, int) else None)
            else:
                st.info("No LGAs match the current filters.")

    st.markdown(
        f"<div class='app-muted'>Model version {_get_version_label()} · Data year {year}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-muted'>Built as part of ongoing research in health data science.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
