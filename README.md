# Health Desert Scorer (Nigeria)

A web-based planning tool to identify Local Government Areas (LGAs) with higher barriers to healthcare access.

**Live app:** https://bashir-healthdesert.streamlit.app/  
**Live demo video** https://youtu.be/_esm4MD74Wg
**Website:** https://www.bashir.bio

---

## Why this tool exists
Healthcare outcomes are strongly influenced by access barriers, including distance to care, facility availability, and connectivity constraints. This tool supports evidence-informed planning at LGA level.

---

## Who can use this
- Frontline health teams
- NGO program teams
- Government policy units
- Donors and implementation partners
- Researchers and students

---

## Quick start
1. Open the app: https://bashir-healthdesert.streamlit.app/
2. Select a **State** (or keep **All Nigeria**).
3. Select a **Year**.
4. Use the left panel to search an LGA and set a focus:
   - All risk
   - Child mortality
   - Facility access
   - Connectivity
   - 5km coverage
5. Review map layers (**Risk score**, **Facilities**, **Connectivity**) and the highest-need list.

---

## Key definitions
- **Health desert:** area where people face stronger practical barriers to care.
- **Risk score:** relative planning score (0-10), not a diagnosis.
- **Highest-need list:** ranked LGAs under current filters.
- **Connectivity layer:** LGA connectivity signal (towers per 10k where available).
- **LGA:** Local Government Area.

---

## How to read the map
- **Green:** relatively lower barriers
- **Yellow/Orange:** moderate barriers
- **Red:** relatively higher barriers

Important: this is a decision-support tool for planning and prioritization. It should not replace field verification.

---

## Responsible use
- Use outputs as a starting point for program design.
- Combine with local context: security, roads, seasonality, staffing, and facility readiness.
- Do not use this tool as the only basis for high-stakes decisions.

---

## Data sources
- DHS (2013, 2018)
- Nigeria Health Facility Registry (NHFR)
- WorldPop
- OpenCellID (for connectivity features when available)

---

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/create_mock_dhs.py
python scripts/download_open_data.py
python -m src.data.build_features
make build-data
make validate-gold
streamlit run app/app.py
```

---

## Data protection and limits
- Repository artifacts are aggregated outputs.
- Do not commit restricted DHS microdata or sensitive coordinates.
- Results should be validated before policy or funding commitments.

---

## Contact
For collaboration or implementation support: https://www.bashir.bio
