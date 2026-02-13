"""Contract validation for app-ready gold outputs."""

from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "data" / "gold"


RISK_REQUIRED = {
    "lga_id",
    "lga_name",
    "state_id",
    "state_name",
    "year",
    "risk_score_total",
    "risk_score_facility_access",
    "risk_score_connectivity",
    "risk_score_mortality",
    "confidence_pct",
    "confidence_reason_codes",
    "model_version",
    "estimate_as_of",
}

EXPLAIN_REQUIRED = {"lga_id", "year", "primary_barriers", "recommendation"}

STATE_REQUIRED = {
    "state_id",
    "state_name",
    "avg_risk",
    "median_risk",
    "max_risk",
    "high_risk_lga_count",
    "lga_count",
    "total_population",
}


def _require_cols(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"{label}: missing required columns: {sorted(missing)}")


def main() -> None:
    risk_path = GOLD_DIR / "gold_lga_risk.csv"
    explain_path = GOLD_DIR / "gold_lga_explain.csv"
    rollup_path = GOLD_DIR / "gold_state_rollup.csv"
    export_csv_path = GOLD_DIR / "gold_exports_csv.csv"
    export_geo_path = GOLD_DIR / "gold_exports_geojson.geojson"
    manifest_path = GOLD_DIR / "gold_export_manifest.json"

    for path in [risk_path, explain_path, rollup_path, export_csv_path, export_geo_path, manifest_path]:
        if not path.exists():
            raise AssertionError(f"Missing gold artifact: {path}")

    risk = pd.read_csv(risk_path)
    _require_cols(risk, RISK_REQUIRED, "gold_lga_risk")
    if not risk["risk_score_total"].between(0, 10).all():
        raise AssertionError("risk_score_total must be between 0 and 10")
    if not risk["confidence_pct"].between(0, 100).all():
        raise AssertionError("confidence_pct must be between 0 and 100")
    if risk[["lga_id", "year"]].duplicated().any():
        raise AssertionError("lga_id/year combinations must be unique")
    if risk["risk_score_total"].isna().any():
        raise AssertionError("risk_score_total cannot be null")

    explain = pd.read_csv(explain_path)
    _require_cols(explain, EXPLAIN_REQUIRED, "gold_lga_explain")
    missing_keys = explain.merge(risk[["lga_id", "year"]], on=["lga_id", "year"], how="left", indicator=True)
    if (missing_keys["_merge"] != "both").any():
        raise AssertionError("gold_lga_explain contains rows not present in gold_lga_risk")

    rollup = pd.read_csv(rollup_path)
    _require_cols(rollup, STATE_REQUIRED, "gold_state_rollup")

    export_csv = pd.read_csv(export_csv_path)
    if export_csv.empty:
        raise AssertionError("gold_exports_csv.csv must not be empty")
    if not set(["lga_id", "year"]).issubset(export_csv.columns):
        raise AssertionError("gold_exports_csv.csv must include lga_id and year")

    export_geo = gpd.read_file(export_geo_path)
    if export_geo.empty:
        raise AssertionError("gold_exports_geojson.geojson must not be empty")
    if "geometry" not in export_geo.columns:
        raise AssertionError("gold_exports_geojson.geojson must include geometry")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "generated_at" not in manifest or "rows" not in manifest or "columns" not in manifest:
        raise AssertionError("gold_export_manifest.json missing required keys")

    print("Gold contracts validated")


if __name__ == "__main__":
    main()
