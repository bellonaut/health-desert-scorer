# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [1.4.1] - 2026-03-25

### Added
- Added a standalone FastAPI runtime in `app/api.py` plus service worker assets and PWA icons so the Nigeria UI can run directly at `localhost:8601` without the Streamlit iframe loop.
- Added a reusable `scripts/validate_pwa.ps1` helper for local API plus PWA wiring checks.
- Added a compact methodology panel to the Streamlit host so the embedded app has adjacent model and scope context.

### Changed
- Changed the Streamlit host to render the Nigeria app directly with `st.components.v1.html(...)` instead of writing a temporary served file.
- Changed embedded payload generation so state-filtered views keep nationwide map geometry available for local chip and LGA rehydration flows.
- Changed the standalone PWA shell to load only first-render assets up front, defer noncritical UI work, and use non-blocking font loading.
- Changed app palettes in the Folium and nurse-view surfaces to an Okabe-Ito-style colorblind-safe scale.
- Changed the global landing pages to report three DHS survey waves.
- Changed travel-time generation to support local ORS execution, checkpoint-only rebuilds, and bounded-memory checkpoint consolidation.

### Fixed
- Fixed the standalone chip and polygon selection loop so clicks update URL state, refetch payload data, recolor the map, and refresh the sidebar without a hard reload.
- Fixed the `Data / Map / Print` control alignment relative to the map chip row.
- Fixed the merge reconciliation between the standalone embedded render path and the UI overhaul so the layer strip, depth toggle, and map/table controls render together without duplicates.
- Fixed deployed Nigeria rendering so the app no longer depends on a browser-local temporary file server path.
- Fixed PWA startup performance by deferring compare, tutorial, print, and support UI work until after first paint.

### Verification
- `pytest -q`
- `node tests/accessibility_test.js`
- `python -m streamlit run app/app.py --server.headless true --server.port 8501`
- `node tmp/playwright_probe.js`
- `npx.cmd lighthouse http://127.0.0.1:8601/?pwa=1 --only-categories=performance --output=json --output-path=tmp/lighthouse-pwa.json --chrome-flags="--headless=new --no-sandbox --disable-dev-shm-usage"`

## [1.4.0] - 2026-03-24

### Added
- Added versioned model artifact output for `models/risk_model_v1.4/`, including `metadata.json` and `MODEL_CARD.md`.
- Added a two-stage Nigeria scoring strategy that keeps a shared all-year base model and applies routed `pop_pct_60min` recalibration to 2024 only.
- Added DHS 2024 as a third year across feature generation, silver outputs, gold outputs, and app year selectors.
- Added mock 2024 DHS KR and HR CSV fallbacks for local bootstrap workflows in `scripts/create_mock_dhs.py`.
- Added versioned model artifact output support for `models/risk_model_v1.3/`.
- Added routed travel-time coverage support across the pipeline, including `pop_pct_30min`, `pop_pct_60min`, `pop_pct_120min`, `pop_pct_within_60min`, and `pop_covered_60min`.
- Added `60-min coverage` as an application focus option, detail metric, and export field.
- Added confidence capping and confidence-band presentation in the application payload and UI.
- Added vendored frontend runtime assets and a lightweight localhost file server so the embedded Nigeria app can load Leaflet, Turf, and export helpers reliably.

### Changed
- Updated gold scoring to load `risk_model_v1.4` and request required features from the artifact rather than relying on a hard-coded feature list.
- Updated the Methodology page and embedded UI defaults to surface model version `v1.4`.
- Updated gold scoring to use model version `v1.3`.
- Updated training to use all three DHS years with temporal validation holding out 2024.
- Updated year selectors to `2024`, `2018`, `2013`, and `Both (avg)`.
- Updated `Both` aggregation so the displayed year remains `Both` rather than a numeric average.
- Updated footer metadata to reference DHS 2013/2018/2024 and ORS isochrones.
- Simplified the desktop header and upgraded the comparison modal for policy-facing review workflows.
- Updated the Nigeria embedded UI with `Data / Map / Print` map modes, publishable polygon seams in data view, score-first detail rendering, and a four-step decision workflow tutorial.
- Updated the Nigeria embed path to default to year `2024`, pass through the parent app route, and use lower-resolution boundaries on mobile or initial desktop load.
- Updated gold risk outputs to rank-normalize `risk_score_total` within each year and keep `risk_score` synchronized as the 0-1 derivative.

### Fixed
- Fixed v1.3-style evaluation leakage by replacing the old binary mortality-threshold training objective with an honest continuous two-stage scoring workflow for v1.4.
- Fixed `risk_score_access_60min` so 2013/2018 rows fall back row-wise to `coverage_5km` instead of being treated as zero-access whenever 2024 routed coverage exists.
- Fixed the release rebuild handoff so corrected travel-time coverage reaches `processed`, `bronze`, `silver`, and `gold` consistently before retraining.
- Fixed DHS household recode alias handling for real HR inputs.
- Fixed model serialization so `src.models.score.score_lga(..., version=\"v1.3\")` resolves correctly.
- Fixed confidence display so hotspot and detail payloads no longer expose raw over-precise percentages.
- Fixed state filtering so selecting a state fits to Nigerian state bounds and dims LGAs outside the selected state instead of zooming to an incorrect continental extent.
- Fixed focus chips so map polygon colors now respond to the active risk dimension alongside the ranked list.
- Fixed hotspot and tooltip driver tags so LGAs no longer default to `mortality critical` when facilities, connectivity, or road access are the dominant barrier.
- Fixed hotspot payload aliases and score precision so the embedded Nigeria UI receives `lga_id`, `state_name`, `worst_driver`, and 2-decimal `risk_score` values consistently.
- Fixed embedded payload serialization so NaN and Infinity values are sanitized across GeoJSON, detail payloads, and map values before HTML injection.
- Fixed app-side score drift by re-synchronizing `risk_score_total` with `risk_score` after temperature scaling and by validating the score spread in gold contracts.

### Verification
- `python -m src.models.train_models`
- `python -m src.data.build_features`
- `python -m src.data.migrate_release_data`
- `python -m src.data.build_silver`
- `python -m src.data.build_gold`
- `python -m src.data.validate_gold_contracts`
- `pytest -q`
- `node tests/accessibility_test.js`
- `streamlit run app/app.py --server.headless true`

## [1.3.1] - 2026-02-18

### Fixed
- Hardened payload serialization in `app/bridge.py` so unknown values no longer crash `_json_default`.
- Added field-level serialization diagnostics via `_find_unserializable` in `inject_data_to_html`.
- Expanded `_coerce` to safely normalize opaque numpy/object scalar wrappers and nested containers.
- Added regression coverage for opaque serialization fallbacks in `tests/test_serialization.py`.

### Verification
- `pytest -q tests/test_serialization.py`
- `pytest -q`

## [1.3.0] - 2026-02-18

### Added
- Mobile-first map controls and floating mobile filters/depth toggle.
- Mobile options drawer and mobile state/year selector sync.
- Robust payload serialization support for numpy/pandas-heavy values.
- Fallback UI empty states for missing map payload and empty hotspot filters.
- New `tests/test_serialization.py` coverage for payload JSON safety.

### Changed
- Detail drawer interactions now use delegated event handling.
- Streamlit app data loading moved to explicit cached loader in `app/app.py`.
- Map zoom controls now use thumb-friendly positioning on mobile.
- Mobile compare strip visibility improved so compare is reachable at depth 0.

### Fixed
- Mobile drawer close reliability and backdrop interaction behavior.
- Mobile layout sizing/overflow regressions in map/panel/header stack.
- Mobile onboarding tour start behavior restored for first-time users.

### Verification
- `make build-data`
- `make validate-gold`
- `pytest -q`
- `node --check app/health_desert_ui.js`
