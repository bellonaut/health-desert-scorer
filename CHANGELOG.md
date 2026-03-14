# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Added DHS 2024 as a third year across feature generation, silver outputs, gold outputs, and app year selectors.
- Added mock 2024 DHS KR and HR CSV fallbacks for local bootstrap workflows in `scripts/create_mock_dhs.py`.
- Added versioned model artifact output support for `models/risk_model_v1.3/`.
- Added routed travel-time coverage support across the pipeline, including `pop_pct_30min`, `pop_pct_60min`, `pop_pct_120min`, `pop_pct_within_60min`, and `pop_covered_60min`.
- Added `60-min coverage` as an application focus option, detail metric, and export field.
- Added confidence capping and confidence-band presentation in the application payload and UI.
- Added vendored frontend runtime assets and a lightweight localhost file server so the embedded Nigeria app can load Leaflet, Turf, and export helpers reliably.

### Changed
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
