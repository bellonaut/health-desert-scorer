"""Methodology page for model transparency and ethical framing."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from utils.analytics import log_event

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "risk_model_v1.4"

st.set_page_config(page_title="Methodology - Health Desert Scorer", page_icon="📊", layout="wide")
st.title("📊 Methodology & Data Sources")

st.markdown(
    """
The **Health Desert Risk Score** is a decision-support indicator for LGA-level healthcare access barriers.

This tool supports planning and prioritization. It is not for clinical diagnosis or individual prediction.
"""
)
st.markdown("Website: [www.bashir.bio](https://www.bashir.bio)")

metrics = {"spearman": 0.0, "mae": 0.0, "rmse": 0.0}
metrics_path = MODEL_DIR / "metrics.json"
if metrics_path.exists():
    metrics.update(json.loads(metrics_path.read_text()))

c1, c2, c3 = st.columns(3)
c1.metric("Spearman rho", f"{metrics.get('v1_4', {}).get('spearman', metrics['spearman']):.2f}")
c2.metric("MAE", f"{metrics.get('v1_4', {}).get('mae', metrics['mae']):.2f}")
c3.metric("RMSE", f"{metrics.get('v1_4', {}).get('rmse', metrics['rmse']):.2f}")

st.subheader("Limitations")
st.warning(
    """
- Does not account for insecurity or conflict context
- Does not capture seasonal road accessibility
- Does not measure care quality or staffing adequacy
- Should always be validated with local health stakeholders
"""
)

st.subheader("Model Card")
model_card = MODEL_DIR / "MODEL_CARD.md"
legacy_model_card = MODEL_DIR / "model_card.md"
card_path = model_card if model_card.exists() else legacy_model_card
if card_path.exists():
    st.download_button(
        "Download Model Card",
        card_path.read_text(encoding="utf-8"),
        file_name="health_desert_model_card.md",
        mime="text/markdown",
    )
else:
    st.info("Model card is not available in this deployment.")

# Testing instrumentation
try:
    params = st.query_params
except Exception:  # pragma: no cover
    params = st.experimental_get_query_params()

if "testing" in params:
    session_id = params.get("session")
    persona = params.get("persona", "unknown")
    if isinstance(session_id, list):
        session_id = session_id[-1]
    if isinstance(persona, list):
        persona = persona[-1]
    log_event(session_id=str(session_id) if session_id else None, persona=str(persona), event_type="methodology_open")
