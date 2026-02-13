"""Plain-language glossary for Health Desert Scorer."""

from __future__ import annotations

import streamlit as st

from utils.analytics import log_event

st.set_page_config(page_title="Glossary - Health Desert Scorer", page_icon="📖", layout="wide")
st.title("📖 Glossary of Terms")

st.markdown("Plain-language definitions of all terms used in this tool.")
st.markdown("Website: [www.bashir.bio](https://www.bashir.bio)")

categories = {
    "Risk & Scoring": {
        "Health Desert": (
            "An area where people face significant barriers to accessing healthcare, such as long distances, "
            "few facilities, or weak connectivity. This is about access constraints, not blame."
        ),
        "Risk Score": (
            "A number from 0-10 showing relative healthcare access barriers compared to other LGAs. "
            "Higher scores mean more barriers. This is for planning, not diagnosing individuals."
        ),
        "Confidence Level": (
            "How certain we are about the score, based on data quality. High confidence means stronger data; "
            "low confidence means limited data, so interpret carefully."
        ),
        "Component Score": (
            "Risk broken down by type: facility access, connectivity, or child health outcomes. "
            "Helps explain what drives the overall score."
        ),
    },
    "Geographic Terms": {
        "LGA": "Local Government Area. Nigeria has 774 LGAs across 36 states and the FCT.",
        "State": "Nigeria's second-tier administrative divisions (36 states plus FCT).",
        "Geospatial": "Related to location and geography. This tool uses maps and location data.",
    },
    "Health System Terms": {
        "Facility Density": "Number of health facilities per 100,000 people. Higher usually means better access.",
        "Primary Health Center (PHC)": "Community-level health facility providing basic care and referrals.",
        "Secondary or Tertiary Care": "Specialized hospitals with surgeries, specialists, and advanced equipment.",
        "Travel Distance": "Average distance to reach the nearest health facility.",
        "5km Coverage": "Percentage of population living within 5km of any health facility.",
        "Under-5 Mortality": "Deaths of children under 5 per 1,000 live births.",
    },
    "Connectivity Terms": {
        "3G or 4G Coverage": "Mobile network coverage that supports communication and telehealth.",
        "Network Coverage %": "Percentage of LGA area with mobile signal.",
        "Towers per 10k": "Number of mobile towers per 10,000 people. More towers usually means better coverage.",
    },
    "Data & Methods": {
        "DHS": "Demographic and Health Surveys. Nationally representative household health data.",
        "NHFR": "Nigeria Health Facility Registry. Official database of facilities.",
        "WorldPop": "High-resolution population distribution data.",
        "Machine Learning": "Algorithms that learn patterns from data to make predictions.",
        "SHAP": "A method that explains which factors contributed most to a score.",
        "Cross-Validation": "A way to test model accuracy by training on some data and testing on other data.",
    },
}

for category, terms in categories.items():
    st.subheader(category)
    for term, definition in terms.items():
        with st.expander(term):
            st.write(definition)

st.markdown("---")

st.info(
    """
Still unclear on a term? Share feedback via GitHub issues or email.
"""
)

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
    log_event(session_id=str(session_id) if session_id else None, persona=str(persona), event_type="glossary_open")
