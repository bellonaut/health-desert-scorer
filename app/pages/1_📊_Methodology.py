"""Methodology page for model transparency and ethical framing."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "risk_model_v1.2"

st.set_page_config(page_title="Methodology - Health Desert Scorer", page_icon="📊", layout="wide")
st.title("📊 Methodology & Data Sources")

st.markdown(
    """
The **Health Desert Risk Score** is a decision-support indicator for LGA-level healthcare access barriers.

> ⚠️ This tool supports planning and prioritization. It is **not** for clinical diagnosis or individual prediction.
"""
)

metrics = {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0}
metrics_path = MODEL_DIR / "metrics.json"
if metrics_path.exists():
    metrics.update(json.loads(metrics_path.read_text()))

c1, c2, c3 = st.columns(3)
c1.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("F1 Score", f"{metrics['f1']:.2f}")
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

st.subheader("⚠️ Limitations")
st.warning(
    """
- Does not account for insecurity or conflict context
- Does not capture seasonal road accessibility
- Does not measure care quality or staffing adequacy
- Should always be validated with local health stakeholders
"""
)

st.subheader("📄 Model Card")
model_card = MODEL_DIR / "model_card.md"
if model_card.exists():
    st.download_button(
        "📥 Download Model Card",
        model_card.read_text(encoding="utf-8"),
        file_name="health_desert_model_card.md",
        mime="text/markdown",
    )
else:
    st.info("Model card is not available in this deployment.")
