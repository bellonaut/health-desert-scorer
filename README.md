## Nigeria Health Desert Risk Scorer

This project builds LGA-level health access features for Nigeria, trains risk models, and provides a Streamlit app for exploration. DHS microdata must **never** be committed; only aggregated LGA-level outputs belong in `data/processed/`. DHS cluster locations are displaced and this pipeline respects that by operating on aggregated outputs only.

> Recommended Python: 3.11 or 3.12 on Windows for geospatial dependencies.

### Run instructions

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
