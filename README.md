## Nigeria Health Desert Risk Scorer

Nigeria faces preventable mortality partly because people cannot reach care in time. This project is an evolving research prototype that links maternal/child health outcomes with geographic access, facility density, and basic connectivity to identify LGAs at elevated risk of “health deserts.” It is part of my MSc Data Science work, designed to translate public health intelligence into decision-support signals while remaining grounded in the realities of data quality, displacement, and infrastructure variability. The goal is not a deployed system—it is a research-grade foundation for understanding where distance, scarcity, and connectivity intersect.

> This is a research-driven prototype, not an operational tool. DHS microdata must **never** be committed; only aggregated LGA-level outputs belong in `data/processed/`. DHS cluster locations are displaced and this pipeline respects that by operating on aggregated outputs only.

---

### Problem Statement
Nigeria’s health outcomes vary sharply by geography. Many LGAs face a compounding mix of long travel times, low facility density, and weak connectivity that can delay or prevent care—particularly for maternal and child health. The challenge is to identify these high-risk areas consistently, using transparent, spatially aware methods that can inform planning without overstating precision.

### Why This Matters (Nigeria + global health)
Nigeria accounts for a significant share of global maternal and child mortality, and geographic access remains a critical bottleneck. The same access-and-connectivity dynamics appear across low-resource settings worldwide. A rigorous, Nigeria-focused prototype can serve as a template for how data science and geospatial intelligence support health equity and policy decisions without becoming a “black box.”

### What This Tool Does
This project constructs LGA-level features from population, facility, and connectivity data; trains interpretable and boosted models; and produces a risk score intended for analysis and exploration. The output is a map-ready signal that can support decision-support discussions, not clinical guidance.

### Who This Is For
- **Public health planners** — to prioritize areas where access gaps may align with poorer outcomes.
- **Policy analysts** — to compare risk patterns with existing investments and programs.
- **NGOs** — to target outreach and evaluate geographic coverage.
- **Infrastructure planners** — to surface where health facility planning intersects with road and connectivity gaps.

---

### Data Sources
- **DHS (with approval)** — aggregated outcomes derived from DHS microdata (approval PDF included in repo).
- **WorldPop** — population density and distribution.
- **GRID3 / facility data** — health facility locations and basic attributes.
- **OpenCellID** — proxy measures for connectivity and coverage.

### Methods
- **Feature engineering**: access metrics, facility density, distance-based measures, and population-weighted indicators.
- **Models**: logistic regression for interpretability plus XGBoost for non-linear patterns.
- **Spatial aggregation**: LGA-level summarization aligned with displaced cluster data.
- **Risk modeling**: produces relative risk signals intended for comparison, not diagnosis.

### Ethics & Limitations
- **Aggregated data only**: no individual-level or personally identifiable information.
- **Not clinical guidance**: outputs are for planning and research use.
- **Early-stage prototype**: methods and assumptions are evolving; results must be validated.

---

### Why I Built This
I built this as a systems-thinking exercise in health equity: to link infrastructure, access, and outcomes in a way that can support decision-support conversations. Nigeria’s health gaps are not just clinical; they’re logistical and geographic. This prototype helps me explore how data science can map those constraints and translate them into actionable, evidence-aware signals.

---

### LinkedIn Feature Description
**Headline:** Nigeria Health Desert Risk Scorer — applied ML for public health access risk

**Description:** A research-grade prototype from my MSc Data Science work that combines DHS outcomes with geospatial access and connectivity indicators to surface LGAs at elevated health-access risk. It is designed for decision-support exploration, not deployment, with an emphasis on transparency and policy relevance.

**Impact highlights:**
- Aggregates DHS-derived signals into LGA-level risk features with displacement-aware handling.
- Combines interpretable models and boosted trees for robust, explainable baselines.
- Integrates facility density, distance, and connectivity proxies into a unified access profile.
- Designed for public health and policy analysis rather than clinical prediction.
- Built as an evolving research prototype with clear limitations and ethical safeguards.

---

### Model Card (Summary)
- **Purpose**: identify LGAs with elevated access-related risk for maternal/child health outcomes.
- **Inputs**: aggregated DHS-derived outcomes, facility density metrics, access distances, population weights, and connectivity proxies.
- **Outputs**: LGA-level relative risk scores and model probabilities for comparative analysis.
- **Interpretation notes**: scores are comparative signals; they do not diagnose conditions or imply causality.
- **Limitations**: relies on aggregated data, displaced clusters, and proxy connectivity data; not validated for clinical decision-making.

---

### Real-World Use Cases
- **Resource allocation**: identify LGAs that may need additional outreach or facility support.
- **Outreach targeting**: prioritize community health programs where access constraints are severe.
- **Health infrastructure planning**: inform facility placement or mobile clinic strategies.
- **Telecom + health partnerships**: connect weak connectivity zones with digital health initiatives.
- **Policy analysis**: compare planned investments against risk patterns for equity impacts.

---

### Proof Assets to Add (Credibility Boosters)
1. **Short demo video** — shows the end-to-end pipeline and map exploration in action.
2. **ROC/PR curves** — demonstrates model behavior without overstating performance.
3. **Feature importance plots** — validates that access variables drive risk signals.
4. **Example LGA insights** — a mini case study showing why a specific LGA scores high.
5. **App screenshots** — communicates a real, working prototype and its interface.

---

### Portfolio Positioning Statement
This project shows that I approach data science as a systems problem: combining geospatial data engineering, interpretable modeling, and policy-aware framing to address health equity. It reflects research depth, technical rigor, and a commitment to building decision-support tools that are honest about limitations yet useful for real-world planning.

---

### Portfolio Checklist (10 Practical Steps)
1. Add a short demo video showing data ingestion, modeling, and app exploration.
2. Publish a compact model card with inputs, outputs, and limitations.
3. Include a “results snapshot” page with 2–3 validated insights and caveats.
4. Add ROC/PR curves and calibration plots with plain-language interpretation.
5. Show a feature importance chart that ties results to access constraints.
6. Document data provenance and access limitations (including DHS approval).
7. Add a short “methods & assumptions” section for spatial aggregation decisions.
8. Include a reproducible run script and pinned dependencies for reproducibility.
9. Provide a sample policy brief (1–2 pages) summarizing implications.
10. Add screenshots of the Streamlit interface and map outputs.

---

> Recommended Python: 3.11 or 3.12 on Windows for geospatial dependencies.

### Run instructions

### Shippable quickstart (Mac/Linux)

> This script sets up a local virtualenv, downloads the open datasets, builds features, trains models, and launches the Streamlit app.

```bash
./scripts/run_demo.sh
```

### Windows quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\create_mock_dhs.py
python scripts\download_open_data.py
python -m src.data.build_features
python -m src.models.train_models
streamlit run app\app.py
```

1. **Create mock DHS clusters (optional for demo data)**
   ```bash
   python scripts/create_mock_dhs.py
   ```
2. **Download or place required inputs**
   ```bash
   python scripts/download_open_data.py
   ```
   Place `lga_boundaries.geojson` and `health_facilities.geojson` in `data/raw/`.
3. **Build features**
   ```bash
   python -m src.data.build_features
   ```
4. **Train models**
   ```bash
   python -m src.models.train_models
   ```
5. **Run app**
   ```bash
   streamlit run app/app.py
   ```

### Stage A Baseline (frozen)
- Uses mock DHS-derived labels (demo only); access metrics come from real boundaries/facilities inputs.
- Outputs: `data/processed/lga_features.csv` and `docs/build_features_report.json`.
- Pipeline runner: `python scripts/create_mock_dhs.py`, `python scripts/download_open_data.py`, then `python -m src.data.build_features` (or use `scripts/run_stage_a.ps1` on Windows).
- Intended as a reproducible baseline before any model changes; keep DHS microdata out of the repo.

### DHS export notes
- `scripts/export_dhs_clusters.py` reads BR files with `convert_categoricals=False` because the 2013 BR has duplicate value labels (e.g., `v131`) that crash pandas category conversion.
- Region codes `v024` are mapped via the label dictionary when present (falling back to the fixed 1–6 region code map) so the North/South split works whether `v024` is numeric or labeled.

### RUN ORDER
- [ ] `python scripts/create_mock_dhs.py`
- [ ] `python scripts/download_open_data.py`
- [ ] `python scripts/aggregate_population.py`
- [ ] *(optional, for tower metrics)* `python -m src.data.build_features` will ingest `data/raw/opencellid.csv.gz` if present
- [ ] `python -m src.data.build_features`
- [ ] `python -m src.models.train_models`
- [ ] `streamlit run app/app.py`
