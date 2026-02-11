# Product Audit (Nigeria Health Desert Scorer)

## Audit question
How close is this project to being a single tool usable by **lay Nigerians**, **nurses**, **policymakers**, and **researchers** at the same time?

## Current stage (where you are now)
- The repo is currently a **research prototype with a strong geospatial + feature-engineering core** and an exploratory UI shell.
- The README explicitly frames this as **non-operational** and **research-first**.
- You already ship derived artifacts (`lga_features.csv`, `lga_predictions.csv`) and a custom map-driven frontend.

## What is good (strengths)
1. **Strong ethical framing and data-safety posture**
   - Clear statements that DHS microdata/coordinates must not be committed.
   - LGA aggregation is appropriate for risk communication and privacy protection.

2. **Good modular separation for data and app layers**
   - Data loading/normalization is isolated (`app/data_api.py`).
   - HTML bridge cleanly isolates payload construction from rendering (`app/bridge.py`).
   - Feature-building pipeline is structured into stages (ingest, aggregate, enrich, validate).

3. **Practical output artifacts already present**
   - Two-year LGA-level outputs exist in `data/processed/`, with complete prediction probability rows.
   - This means the project can demonstrate end-to-end value without private DHS files in-repo.

4. **Helpful intent to support reproducibility**
   - Quickstart scripts and doctor diagnostics are present.
   - Tests target critical spatial behavior (coverage bounds, row counts, CRS checks).

## Where it falters (gaps against multi-audience goal)
1. **Audience-product fit is still researcher-heavy**
   - Current framing and naming focus on prototype/research; there is no explicit user journey for each audience.
   - Lay users and frontline nurses need plain-language explanations and “what action now?” outputs, which are not formalized.

2. **Run path/documentation drift**
   - README and scripts reference model training module `src.models.train_models`, but `src/models/` is absent in repository files.
   - Tests expect `models/logreg.pkl` and `models/xgb.pkl`, reinforcing that training artifacts/workflow are incomplete in this tree.

3. **Environment onboarding is brittle**
   - In this environment, `pytest` and `scripts/doctor.py` fail at import time because core deps (`pandas`, `geopandas`, `numpy`) are missing.
   - This is expected in a clean machine, but it highlights that “works first try” depends on full dependency install and system geospatial stack.

4. **Data download automation is mostly manual placeholders**
   - `download_open_data.py` defines sources but URLs are `None`; required inputs still require manual retrieval.
   - That slows adoption by non-technical stakeholders and field teams.

5. **No explicit trust + uncertainty UX for non-research users**
   - Risk score is present, but confidence, uncertainty bands, and decision-safe language are not surfaced as first-class outputs.
   - For policymakers and nurses, this can cause over-interpretation.

6. **Localization and accessibility not yet visible**
   - UI language is currently English-only and visually dense.
   - No role-based mode (citizen/simple vs technical/analyst) is evident.

## Scorecard vs target users
- **Lay Nigerians:** 4/10 (map exists, but too technical and not action-guided yet)
- **Nurses/frontline teams:** 5/10 (relevant indicators, but no clinical workflow framing)
- **Policymakers:** 6/10 (LGA prioritization useful; needs scenario and budget-planning views)
- **Researchers:** 8/10 (strongest current fit)

## Priority roadmap (next 6-8 weeks)
1. **Stabilize runnable baseline**
   - Restore/commit model training module or remove stale references.
   - Make `run_demo.sh` and README fully accurate to current repo state.

2. **Add role-based product modes**
   - Citizen mode: plain language, “nearest options,” “what this means,” high-level risk.
   - Nurse mode: service gap indicators, outreach planning cards.
   - Policy mode: ranked interventions + expected coverage impact.
   - Research mode: full metrics and SHAP/feature detail.

3. **Add uncertainty + safe interpretation layer**
   - Display confidence/quality flags by LGA.
   - Add caution labels where signal quality is low.

4. **Lower deployment friction**
   - Provide prebuilt demo dataset bundle and one-click launch instructions.
   - Optionally ship Docker dev image for geospatial dependencies.

5. **Operationalize evidence quality**
   - Data freshness metadata (per source, per year).
   - Validation dashboard (coverage of joins, missingness, outlier LGAs).

## Bottom line
This is a **promising, credible research foundation** with good technical bones and responsible ethics. To serve all four audiences at once, it now needs a **productization pass**: role-based UX, uncertainty communication, and zero-friction run/deploy consistency.
