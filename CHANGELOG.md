# Changelog

All notable changes to this project are documented in this file.

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

