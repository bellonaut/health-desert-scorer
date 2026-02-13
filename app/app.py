"""Streamlit entry point that embeds the bespoke HTML frontend."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from bridge import render_embedded_app
from data_api import load_backend_data, latest_year
from utils.analytics import log_event
from utils.error_handler import safe_execute, show_system_status

# Defaults kept in session_state for two-way sync with the embedded UI
SESSION_DEFAULTS: Mapping[str, Any] = {
    "hd_state_filter": "All Nigeria",
    "hd_depth": 0,
    "hd_focus": "All risk",
    "hd_selected_lga": None,
    "hd_compare_lgas": [],
    "hd_year": "2018",
    "hd_is_mobile": False,
    "hd_testing_mode": False,
    "hd_test_persona": "unknown",
    "hd_test_session": None,
}

PAGES_DIR = Path(__file__).resolve().parent / "pages"
METHOD_ICON = "\U0001F4CA"
GLOSSARY_ICON = "\U0001F4D6"


def _page_path(suffix: str) -> str:
    matches = sorted(PAGES_DIR.glob(f"*{suffix}"))
    if matches:
        return f"pages/{matches[0].name}"
    return f"pages/{suffix}"


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
    _maybe_set("mobile", "hd_is_mobile", lambda val: str(val).lower() in {"1", "true", "yes"})

    if "compare" in params:
        cmp_val = params["compare"]
        if isinstance(cmp_val, list):
            cmp_val = cmp_val[-1]
        st.session_state["hd_compare_lgas"] = [uid for uid in str(cmp_val).split(",") if uid]

    _maybe_set("year", "hd_year")

    if "testing" in params:
        st.session_state["hd_testing_mode"] = str(params["testing"]).lower() in {"1", "true", "yes"}
    if "persona" in params:
        persona = params["persona"]
        if isinstance(persona, list):
            persona = persona[-1]
        st.session_state["hd_test_persona"] = str(persona)
    if "session" in params:
        session = params["session"]
        if isinstance(session, list):
            session = session[-1]
        st.session_state["hd_test_session"] = str(session)
    elif st.session_state.get("hd_testing_mode") and not st.session_state.get("hd_test_session"):
        st.session_state["hd_test_session"] = uuid.uuid4().hex

    if "evt" in params and st.session_state.get("hd_testing_mode"):
        evt_raw = params["evt"]
        if isinstance(evt_raw, list):
            evt_raw = evt_raw[-1]
        last_evt = st.session_state.get("hd_last_evt")
        if evt_raw and evt_raw != last_evt:
            try:
                payload = json.loads(str(evt_raw))
            except Exception:
                payload = {"type": "unknown", "details": {"raw": str(evt_raw)}}
            log_event(
                session_id=st.session_state.get("hd_test_session"),
                persona=st.session_state.get("hd_test_persona", "unknown"),
                event_type=payload.get("type", "unknown"),
                details=payload.get("details", {}),
            )
            st.session_state["hd_last_evt"] = evt_raw


def main() -> None:
    st.set_page_config(
        page_title="HEALTHDESERT \u00b7 NG",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )
    _init_session_state()
    _hydrate_from_query_params()

    with st.sidebar:
        st.markdown("### Transparency")
        st.page_link(_page_path("Methodology.py"), label=f"{METHOD_ICON} Methodology")
        st.page_link(_page_path("Glossary.py"), label=f"{GLOSSARY_ICON} Glossary")

    @safe_execute("Load backend data")
    def _load() -> tuple[Any, Any]:
        return load_backend_data(
            source_mode="gold_first",
            boundary_resolution="auto",
            is_mobile=bool(st.session_state.get("hd_is_mobile")),
            zoom=None,
        )

    data = _load()
    if data is None:
        st.stop()

    geo_df, shap_df = data
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
    show_system_status(
        data_last_updated=geo_df.attrs.get("data_last_updated"),
        model_version=geo_df.attrs.get("model_version"),
    )


if __name__ == "__main__":
    main()
