"""Build validated silver-layer datasets from raw/processed sources."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SILVER = ROOT / "data" / "silver"


def _slug(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)


def _make_lga_id(lga_name: pd.Series, state_name: pd.Series | None = None) -> pd.Series:
    lga_slug = _slug(lga_name)
    if state_name is None:
        return lga_slug
    state_slug = _slug(state_name)
    state_slug = state_slug.where(state_name.notna() & state_name.astype(str).str.strip().ne(""), "")
    return (state_slug + "__" + lga_slug).where(state_slug.ne(""), lga_slug)


def _pick_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {[str(c) for c in candidates]}")


def build_silver_lga_master() -> gpd.GeoDataFrame:
    SILVER.mkdir(parents=True, exist_ok=True)
    boundary_path = _pick_path(
        BRONZE / "boundaries" / "lga_boundaries.geojson",
        RAW / "lga_boundaries.geojson",
    )
    lgas = gpd.read_file(boundary_path)
    lgas = lgas.rename(columns={"lganame": "lga_name", "statename": "state_name", "uniq_id": "lga_uid"})
    lgas["lga_name"] = lgas["lga_name"].astype(str).str.strip()
    lgas["state_name"] = lgas["state_name"].astype(str).str.strip()
    lgas["lga_id"] = _make_lga_id(lgas["lga_name"], lgas["state_name"])
    lgas["state_id"] = _slug(lgas["state_name"])
    lgas["geometry"] = lgas.geometry.buffer(0)
    lgas = lgas[lgas.geometry.notna()].copy()
    assert lgas["lga_id"].is_unique, "LGA IDs must be unique"
    assert lgas.geometry.is_valid.all(), "All geometries must be valid"
    lgas.to_file(SILVER / "lga_master.geojson", driver="GeoJSON")
    return lgas


def build_silver_metrics() -> pd.DataFrame:
    features_path = _pick_path(
        BRONZE / "derived" / "lga_features.csv",
        PROCESSED / "lga_features.csv",
    )
    features = pd.read_csv(features_path)
    preds_path = _pick_path(
        BRONZE / "derived" / "lga_predictions.csv",
        PROCESSED / "lga_predictions.csv",
    )
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()

    features = features.rename(columns={"towers_per_10k_pop": "towers_per_10k"})
    if not preds.empty:
        if "risk_score" not in preds.columns and "risk_prob" in preds.columns:
            preds = preds.rename(columns={"risk_prob": "risk_score"})
        if "risk_score" in preds.columns:
            preds = (
                preds.groupby(["lga_name", "year"], as_index=False)["risk_score"].mean()
                if {"lga_name", "year"} <= set(preds.columns)
                else preds
            )
            features = features.merge(preds[["lga_name", "year", "risk_score"]], on=["lga_name", "year"], how="left")

    features["lga_id"] = _make_lga_id(features["lga_name"], features["state_name"])
    features["state_id"] = _slug(features["state_name"])
    features["coverage_3g_pct"] = pd.NA
    features["coverage_4g_pct"] = pd.NA
    features["data_quality_flag"] = "ok"
    features.loc[features["facilities_per_10k"].fillna(0) <= 0, "data_quality_flag"] = "no_facilities"
    features.to_csv(SILVER / "lga_metrics.csv", index=False)
    return features


def build_silver_data_quality_profiles() -> pd.DataFrame:
    metrics = pd.read_csv(SILVER / "lga_metrics.csv")

    issues = []
    for row in metrics.itertuples(index=False):
        codes: list[str] = []
        if pd.isna(row.u5mr_mean):
            codes.append("no_dhs_data")
        elif float(row.u5mr_mean) <= 0:
            codes.append("low_dhs_sample")
        if pd.isna(row.facilities_per_10k) or float(row.facilities_per_10k) <= 0:
            codes.append("no_facility_data")

        overall = "high"
        if len(codes) == 1:
            overall = "medium"
        elif len(codes) >= 2:
            overall = "low"

        issues.append(
            {
                "lga_id": row.lga_id,
                "year": row.year,
                "confidence_issues": "|".join(codes) if codes else "none",
                "overall_quality": overall,
            }
        )

    quality = pd.DataFrame(issues)
    quality.to_csv(SILVER / "data_quality_profiles.csv", index=False)
    return quality


def build_silver_boundaries() -> None:
    lgas = gpd.read_file(SILVER / "lga_master.geojson")

    hi = lgas.copy()
    hi["geometry"] = hi["geometry"].simplify(0.005, preserve_topology=True)
    hi.to_file(SILVER / "lga_boundaries_hi.geojson", driver="GeoJSON")

    med = lgas.copy()
    med["geometry"] = med["geometry"].simplify(0.01, preserve_topology=True)
    med.to_file(SILVER / "lga_boundaries_med.geojson", driver="GeoJSON")

    low = lgas.copy()
    low["geometry"] = low["geometry"].simplify(0.02, preserve_topology=True)
    low.to_file(SILVER / "lga_boundaries_lo.geojson", driver="GeoJSON")

    if "state_id" in lgas.columns:
        states = lgas.dissolve(by="state_id", aggfunc="first")
    elif "state_name" in lgas.columns:
        states = lgas.dissolve(by="state_name", aggfunc="first")
    else:
        states = lgas.copy()
    states["geometry"] = states["geometry"].simplify(0.03, preserve_topology=True)
    states.to_file(SILVER / "state_boundaries.geojson", driver="GeoJSON")


def main() -> None:
    build_silver_lga_master()
    build_silver_metrics()
    build_silver_data_quality_profiles()
    build_silver_boundaries()
    print("Silver data build complete")


if __name__ == "__main__":
    main()
