# AGENTS.md

## Release Governance
- Treat repository release versions and model artifact versions as separate concerns.
- Use SemVer for repository releases, for example `1.4.0`.
- Keep model artifact versions explicit and scoped, for example `risk_model_v1.3`.
- Do not imply a repository release version from a model artifact version.

## Push Readiness
- Before any push requested by the user, run `git status --short` and identify unrelated tracked or untracked changes.
- Do not bundle unrelated work into the same push unless the user explicitly asks for it.
- Exclude preview files, scratch outputs, and local-only artifacts unless they are part of the requested release.
- Update `CHANGELOG.md` before a release push or release branch handoff.
- Use Conventional Commits for commit titles.
- Prefer release work on `release/vX.Y.Z-*` branches and feature work on `feat/*` branches.

## Required Validation Before Push
- For data pipeline changes:
  - `python -m src.data.build_features`
  - `python -m src.data.migrate_release_data`
  - `python -m src.data.build_silver`
  - `python -m src.data.build_gold`
  - `python -m src.data.validate_gold_contracts`
- For test validation:
  - `pytest -q`
- For frontend or UX changes:
  - `node tests/accessibility_test.js`
  - `streamlit run app/app.py --server.headless true`

## Release Notes Standard
- Record release notes in `CHANGELOG.md` using `Added`, `Changed`, `Fixed`, and `Verification` sections.
- In push or handoff summaries, report:
  - branch name
  - commit title or PR title
  - validation commands run
  - known limitations or data caveats

## Current Release Track
- The next feature release after `1.3.1` should be treated as `1.4.0` unless the scope is reduced before push.
