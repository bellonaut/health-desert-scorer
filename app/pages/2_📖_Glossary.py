"""Plain-language glossary for Health Desert Scorer."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Glossary - Health Desert Scorer", page_icon="📖", layout="wide")
st.title("📖 Glossary")

terms = {
    "Health Desert": "An area facing systemic barriers to healthcare access, such as distance, facility scarcity, or weak connectivity.",
    "Risk Score": "A 0-10 relative score indicating access barriers across LGAs. Higher scores mean greater barriers.",
    "Confidence Level": "An estimate (0-100%) of data confidence for the score, based on missingness and data quality checks.",
    "LGA": "Local Government Area, Nigeria's third-level administrative division.",
    "DHS": "Demographic and Health Survey, a nationally representative household survey.",
}

for term, definition in terms.items():
    with st.expander(term):
        st.write(definition)
