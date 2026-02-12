"""Streamlit entry point that embeds the bespoke HTML frontend."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from bridge import render_embedded_app
from data_api import load_backend_data, latest_year

# Defaults kept in session_state for two-way sync with the embedded UI
SESSION_DEFAULTS: Mapping[str, Any] = {
    "hd_state_filter": "All Nigeria",
    "hd_depth": 0,
    "hd_focus": "All risk",
    "hd_selected_lga": None,
    "hd_compare_lgas": [],
    "hd_year": "2018",
}


def _init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _hydrate_from_query_params() -> None:
    """Pull incoming query params set by the JS layer into session_state."""
    try:
        params = st.query_params  # Streamlit 1.30+
    except Exception:  # pragma: no cover - fallback for older versions
        params = st.experimental_get_query_params()

    def _maybe_set(name: str, target: str, cast=None) -> None:
        if name not in params:
            return
        value = params[name]
        # Streamlit may return a list; keep the last value for idempotency
        if isinstance(value, list):
            value = value[-1]
        if cast:
            try:
                value = cast(value)
            except Exception:
                return
        st.session_state[target] = value

    _maybe_set("state", "hd_state_filter")
    _maybe_set("focus", "hd_focus")
    _maybe_set("depth", "hd_depth", int)
    _maybe_set("lga", "hd_selected_lga")
    if "compare" in params:
        cmp_val = params["compare"]
        if isinstance(cmp_val, list):
            cmp_val = cmp_val[-1]
        st.session_state["hd_compare_lgas"] = [uid for uid in str(cmp_val).split(",") if uid]
    _maybe_set("year", "hd_year")


def main() -> None:
    st.set_page_config(
        page_title="HEALTHDESERT · NG",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )
    _init_session_state()
    _hydrate_from_query_params()

    with st.sidebar:
        st.markdown("### Transparency")
        st.page_link("pages/1_📊_Methodology.py", label="📊 Methodology")
        st.page_link("pages/2_📖_Glossary.py", label="📖 Glossary")

    geo_df, shap_df = load_backend_data()
    if st.session_state.get("hd_year") is None:
        st.session_state["hd_year"] = latest_year(geo_df)

    # Full-bleed: remove Streamlit padding/chrome and force iframe to viewport size
    st.markdown(
        """
<style>
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
    }
    .block-container, .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        width: 100vw;
        height: 100vh;
        position: fixed;
        inset: 0;
    }
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }
    .main, [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"], .element-container {
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stHeader"] { display: none !important; height: 0 !important; }
    [data-testid="stToolbar"] { display: none !important; }
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    iframe[title="st.components.v1.html"] {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
    }
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<style>html, body {overflow:hidden !important;}</style>", unsafe_allow_html=True)

    render_embedded_app(geo_df, shap_df, st.session_state)


if __name__ == "__main__":
    main()
