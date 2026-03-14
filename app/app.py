"""Streamlit entry point that embeds the bespoke HTML frontend."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import streamlit as st

from bridge import render_embedded_app
from data_api import latest_year
from utils.analytics import log_event
from utils.error_handler import safe_execute, show_system_status

# Defaults kept in session_state for two-way sync with the embedded UI
SESSION_DEFAULTS: Mapping[str, Any] = {
    "hd_state_filter": "All Nigeria",
    "hd_depth": 0,
    "hd_focus": "All risk",
    "hd_selected_lga": None,
    "hd_compare_lgas": [],
    "hd_year": "2024",
    "hd_is_mobile": False,
    "hd_testing_mode": False,
    "hd_test_persona": "unknown",
    "hd_test_session": None,
    "hd_parent_app_path": "/?app=ng",
}

PAGES_DIR = Path(__file__).resolve().parent / "pages"
GLOBAL_HTML_PATH = Path(__file__).resolve().parent / "static" / "global.html"
SHAP_VALUES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "shap_values.csv"
METHOD_ICON = "\U0001F4CA"
GLOSSARY_ICON = "\U0001F4D6"
GLOBAL_ICON = "\U0001F30D"
NG_ROUTE_HINT_KEYS = {
    "state",
    "focus",
    "depth",
    "lga",
    "compare",
    "year",
    "mobile",
    "testing",
    "persona",
    "session",
    "evt",
}


def _page_path(suffix: str) -> str:
    matches = sorted(PAGES_DIR.glob(f"*{suffix}"))
    if matches:
        return f"pages/{matches[0].name}"
    return f"pages/{suffix}"


def _get_query_params() -> Mapping[str, Any]:
    try:
        return st.query_params  # Streamlit 1.30+
    except Exception:  # pragma: no cover - fallback for older versions
        return st.experimental_get_query_params()


def _last_param_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[-1]
    return str(value)


def _resolve_app_route() -> str:
    params = _get_query_params()

    for key in ("app", "page", "route"):
        if key not in params:
            continue
        route = _last_param_value(params[key]).strip().lower()
        if route in {"ng", "nigeria", "app", "dashboard"}:
            return "ng"
        if route in {"global", "landing", "home", "index"}:
            return "global"

    if any(key in params for key in NG_ROUTE_HINT_KEYS):
        return "ng"
    return "global"


def _init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _file_mtime(path: Path) -> float:
    """Cache key helper: change when the file appears or is updated."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0


def _build_parent_app_path(route: str) -> str:
    params = _get_query_params()
    route_params: dict[str, str] = {}
    if route == "ng":
        route_params["app"] = "ng"
    elif route == "global":
        route_params["app"] = "global"

    for key in ("testing", "persona", "session"):
        if key in params:
            route_params[key] = _last_param_value(params[key])

    query = urlencode(route_params)
    return f"/?{query}" if query else "/"


def _hydrate_from_query_params() -> None:
    """Pull incoming query params set by the JS layer into session_state."""
    params = _get_query_params()

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

    # Migrate old 3-level depth values to the current 2-level scheme:
    # 0 = overview, 1 = research.
    raw_depth = st.session_state.get("hd_depth", 0)
    try:
        depth_value = int(raw_depth)
    except Exception:
        depth_value = 0
    if depth_value >= 2:
        depth_value = 1
    if depth_value < 0:
        depth_value = 0
    st.session_state["hd_depth"] = depth_value

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


def _inject_full_bleed_styles() -> None:
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
        background: #090b10 !important;
        color: #e8eaf0 !important;
    }
    .block-container, .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        background: #090b10 !important;
    }
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        width: 100vw;
        height: 100vh;
        height: 100dvh;
        min-height: -webkit-fill-available;
        position: fixed;
        inset: 0;
        background: #090b10 !important;
    }
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
        background: #090b10 !important;
    }
    .main, [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"], .element-container {
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        overflow: hidden !important;
        background: #090b10 !important;
    }
    [data-testid="stHeader"] { display: none !important; height: 0 !important; }
    [data-testid="stToolbar"] { display: none !important; }
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    iframe[title="st.components.v1.html"],
    iframe[title="st.iframe"] {
        width: 100vw !important;
        height: 100vh !important;
        height: 100dvh !important;
        min-height: -webkit-fill-available !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
    }
    [data-testid="stCustomComponentV1"] {
        height: 100vh !important;
        height: 100dvh !important;
        min-height: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stElementContainer"]:has(> iframe[title="st.components.v1.html"]),
    [data-testid="stElementContainer"]:has(> iframe[title="st.iframe"]) {
        height: 100vh !important;
        height: 100dvh !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    #hd-boot-overlay {
        position: fixed;
        inset: 0;
        z-index: 2147483000;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(circle at 50% 45%, rgba(30, 41, 59, 0.35), rgba(9, 11, 16, 0.96)),
            #090b10;
        transition: opacity 0.3s ease, visibility 0.3s ease;
        opacity: 1;
        visibility: visible;
        animation: hdBootFailSafeHide 0s linear 14s forwards;
    }
    #hd-boot-overlay.is-hidden {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
    }
    .hd-boot-inner {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
    }
    .hd-boot-ring {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-top-color: #f97316;
        animation: hdBootSpin 0.9s linear infinite;
    }
    .hd-boot-label {
        font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 11px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #9ca3af;
    }
    @keyframes hdBootSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes hdBootFailSafeHide {
        to {
            opacity: 0;
            visibility: hidden;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .hd-boot-ring { animation: none; }
    }
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<style>html, body {overflow:hidden !important;}</style>", unsafe_allow_html=True)


def _render_boot_overlay() -> None:
    st.markdown(
        """
<div id="hd-boot-overlay" role="status" aria-live="polite" aria-label="Loading">
  <div class="hd-boot-inner">
    <div class="hd-boot-ring" aria-hidden="true"></div>
    <div class="hd-boot-label">Loading HEALTHDESERT</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_global_landing() -> None:
    if not GLOBAL_HTML_PATH.exists():
        st.error(f"Global landing page not found: `{GLOBAL_HTML_PATH}`")
        st.stop()

    html = GLOBAL_HTML_PATH.read_text(encoding="utf-8")
    st.components.v1.html(html, height=10000, scrolling=False)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load(
    source_mode: str,
    boundary_resolution: str,
    is_mobile: bool,
    zoom: float | None,
    shap_values_mtime: float,
) -> tuple[Any, Any]:
    # `shap_values_mtime` is intentionally unused except as a cache key input.
    from data_api import load_backend_data

    return load_backend_data(
        source_mode=source_mode,
        boundary_resolution=boundary_resolution,
        is_mobile=is_mobile,
        zoom=zoom,
    )


def main() -> None:
    st.set_page_config(
        page_title="HEALTHDESERT",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )

    route = _resolve_app_route()
    _init_session_state()
    _inject_full_bleed_styles()
    _render_boot_overlay()

    if route == "global":
        _render_global_landing()
        return

    _hydrate_from_query_params()
    st.session_state["hd_parent_app_path"] = _build_parent_app_path(route)

    with st.sidebar:
        st.markdown("### Transparency")
        st.markdown(f"[{GLOBAL_ICON} Global platform](?app=global)")
        st.page_link(_page_path("Methodology.py"), label=f"{METHOD_ICON} Methodology")
        st.page_link(_page_path("Glossary.py"), label=f"{GLOSSARY_ICON} Glossary")

    is_mobile = bool(st.session_state.get("hd_is_mobile"))

    @safe_execute("Load backend data")
    def _load() -> tuple[Any, Any]:
        return _cached_load(
            source_mode="gold_first",
            boundary_resolution="low" if is_mobile else "medium",
            is_mobile=is_mobile,
            zoom=None,
            shap_values_mtime=_file_mtime(SHAP_VALUES_PATH),
        )

    data = _load()
    if data is None:
        st.stop()

    geo_df, shap_df = data
    if st.session_state.get("hd_year") is None:
        st.session_state["hd_year"] = latest_year(geo_df)

    html = render_embedded_app(geo_df, shap_df, st.session_state)
    st.components.v1.html(html, height=10000, scrolling=False)
    show_system_status(
        data_last_updated=geo_df.attrs.get("data_last_updated"),
        model_version=geo_df.attrs.get("model_version"),
    )


if __name__ == "__main__":
    main()
