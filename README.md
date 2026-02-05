## Nigeria Health Desert Risk Scorer

This project builds LGA-level health access features for Nigeria, trains risk models, and provides a Streamlit app for exploration. DHS microdata must **never** be committed; only aggregated LGA-level outputs belong in `data/processed/`. DHS cluster locations are displaced and this pipeline respects that by operating on aggregated outputs only.

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

### RUN ORDER
- [ ] `python scripts/create_mock_dhs.py`
- [ ] `python scripts/download_open_data.py`
- [ ] `python -m src.data.build_features`
- [ ] `python -m src.models.train_models`
- [ ] `streamlit run app/app.py`
