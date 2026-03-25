"""Build gold-layer, app-ready decision tables."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.models.score import get_required_features, score_lga

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
MODEL_VERSION = "v1.4"


def _rank_normalize_scores(df: pd.DataFrame, score_col: str = "risk_score_total") -> pd.DataFrame:
    """
    Spread raw model outputs within each year using percentile ranks.

    The gold-layer contract in this repo treats `risk_score_total` as the
    authoritative 0-10 score and `risk_score` as the derived 0-1 view used by
    some app paths, so both columns are recomputed together after ranking.
    """
    normalized = df.copy()
    if score_col not in normalized.columns:
        raise KeyError(f"Missing score column for normalization: {score_col}")

    normalized[score_col] = pd.to_numeric(normalized[score_col], errors="coerce").astype("float64")
    for year in sorted(pd.Series(normalized["year"]).dropna().unique().tolist()):
        mask = (normalized["year"] == year) & normalized[score_col].notna()
        raw = normalized.loc[mask, score_col].to_numpy(dtype=float)
        n = len(raw)
        if n < 2:
            continue
        ranks = rankdata(raw, method="average")
        scaled = (ranks - 1) / (n - 1) * 10.0
        normalized.loc[mask, score_col] = np.round(scaled, 4)

    normalized["risk_score"] = np.round(pd.to_numeric(normalized[score_col], errors="coerce") / 10.0, 6)
    normalized["risk_score_total"] = np.round(normalized["risk_score"] * 10.0, 4)
    return normalized


def calculate_confidence(confidence_issues: str) -> int:
    confidence = 100
    issues = confidence_issues.split("|") if confidence_issues and confidence_issues != "none" else []
    for issue in issues:
        if issue == "no_dhs_data":
            confidence -= 50
        elif issue == "low_dhs_sample":
            confidence -= 20
        elif issue == "no_facility_data":
            confidence -= 30
    return max(0, confidence)


def _component_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["risk_score_facility_access"] = (10 - (df["facilities_per_10k"].fillna(0).clip(0, 10))).clip(0, 10)
    out["risk_score_connectivity"] = (10 - (df["coverage_5km"].fillna(0) / 10)).clip(0, 10)
    # Use routed 60-min coverage when present, otherwise fall back row-wise to straight-line 5km coverage.
    routed = pd.to_numeric(df["pop_pct_60min"], errors="coerce") if "pop_pct_60min" in df.columns else pd.Series(np.nan, index=df.index)
    fallback_cov = pd.to_numeric(df["coverage_5km"], errors="coerce").fillna(0)
    effective_cov = routed.where(routed.notna(), fallback_cov)
    out["risk_score_access_60min"] = (10 - (effective_cov.fillna(0) / 10)).clip(0, 10)
    out["risk_score_mortality"] = (df["u5mr_mean"].fillna(0) / 20).clip(0, 10)
    return out


def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "lga_id" not in df.columns and "lga_name" in df.columns:
        lga_slug = df["lga_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        if "state_name" in df.columns:
            state_slug = df["state_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            state_slug = state_slug.where(df["state_name"].notna() & df["state_name"].astype(str).str.strip().ne(""), "")
            df["lga_id"] = (state_slug + "__" + lga_slug).where(state_slug.ne(""), lga_slug)
        else:
            df["lga_id"] = lga_slug
    if "state_id" not in df.columns and "state_name" in df.columns:
        df["state_id"] = df["state_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df


def build_gold_lga_risk() -> pd.DataFrame:
    GOLD.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SILVER / "lga_metrics.csv")
    quality = pd.read_csv(SILVER / "data_quality_profiles.csv")

    metrics = _ensure_ids(metrics)
    df = metrics.merge(quality, on=["lga_id", "year"], how="left")
    # Carry travel time columns forward if present in silver metrics
    _tt_cols = [c for c in ["pop_pct_30min", "pop_pct_60min", "pop_pct_within_60min"] if c in df.columns]
    if not _tt_cols:
        for col in ["pop_pct_30min", "pop_pct_60min", "pop_pct_within_60min"]:
            df[col] = float("nan")

    try:
        feature_cols = get_required_features(MODEL_VERSION)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        model_scores = score_lga(df[feature_cols], version=MODEL_VERSION)
        df["risk_score_total"] = pd.to_numeric(model_scores["risk_score_total"], errors="coerce")
        df["model_version"] = model_scores["model_version"]
    except Exception:
        # Deterministic fallback preserves reproducibility even without a local binary artifact.
        # Use routed 60-min coverage if available, else straight-line 5km
        _cov = (
            df["pop_pct_60min"].fillna(df["coverage_5km"].fillna(0))
            if "pop_pct_60min" in df.columns
            else df["coverage_5km"].fillna(0)
        )
        df["risk_score_total"] = (
            (10 - df["facilities_per_10k"].fillna(0).clip(0, 10)) * 0.35
            + (df["avg_distance_km"].fillna(0).clip(0, 20) / 2) * 0.20
            + (df["u5mr_mean"].fillna(0).clip(0, 200) / 20) * 0.25
            + (10 - (_cov.clip(0, 100) / 10)) * 0.20
        )
        df["model_version"] = f"{MODEL_VERSION}-fallback"

    components = _component_scores(df)
    df = pd.concat([df, components], axis=1)
    df["risk_score_total"] = df["risk_score_total"].clip(0, 10)
    df["confidence_reason_codes"] = df["confidence_issues"].fillna("none")
    df["confidence_pct"] = df["confidence_reason_codes"].apply(calculate_confidence)
    df["estimate_as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = _rank_normalize_scores(df)

    assert df["risk_score_total"].between(0, 10).all(), "Risk scores must be 0-10"
    assert not df["risk_score_total"].isna().any(), "No null risk scores allowed"

    df.to_csv(GOLD / "gold_lga_risk.csv", index=False)
    return df


def build_gold_lga_explain() -> pd.DataFrame:
    risk_data = pd.read_csv(GOLD / "gold_lga_risk.csv")
    explanations = []
    for row in risk_data.itertuples(index=False):
        factors = []
        if row.facilities_per_10k < 0.5:
            factors.append("Low facility density")
        if row.coverage_5km < 20:
            factors.append("Limited facility proximity coverage")
        if row.u5mr_mean > 120:
            factors.append("Elevated under-5 mortality indicators")
        explanations.append(
            {
                "lga_id": row.lga_id,
                "year": row.year,
                "primary_barriers": " | ".join(factors) if factors else "Multiple moderate access barriers",
                "recommendation": (
                    "Priority for mobile clinic outreach and infrastructure support"
                    if row.risk_score_total >= 7
                    else "Monitor barriers and coordinate targeted interventions"
                    if row.risk_score_total >= 4
                    else "Maintain services and improve care quality"
                ),
            }
        )
    explain_df = pd.DataFrame(explanations)
    explain_df.to_csv(GOLD / "gold_lga_explain.csv", index=False)
    return explain_df


def build_gold_state_rollup() -> pd.DataFrame:
    risk = pd.read_csv(GOLD / "gold_lga_risk.csv")
    risk = _ensure_ids(risk)
    grouped = risk.groupby(["state_id", "state_name"], dropna=False)
    rollup = grouped.agg(
        avg_risk=("risk_score_total", "mean"),
        median_risk=("risk_score_total", "median"),
        max_risk=("risk_score_total", "max"),
        lga_count=("lga_id", "count"),
        total_population=("population", "sum"),
    ).reset_index()
    high_risk = risk[risk["risk_score_total"] >= 7].groupby(["state_id", "state_name"]).size().reset_index(name="high_risk_lga_count")
    rollup = rollup.merge(high_risk, on=["state_id", "state_name"], how="left")
    rollup["high_risk_lga_count"] = rollup["high_risk_lga_count"].fillna(0).astype(int)
    rollup.to_csv(GOLD / "gold_state_rollup.csv", index=False)
    return rollup


def build_gold_exports() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    risk = pd.read_csv(GOLD / "gold_lga_risk.csv")
    explain = pd.read_csv(GOLD / "gold_lga_explain.csv")
    lgas = gpd.read_file(SILVER / "lga_master.geojson")

    export_df = risk.merge(explain, on=["lga_id", "year"], how="left")
    export_df.to_csv(GOLD / "gold_exports_csv.csv", index=False)

    export_geo = lgas.merge(export_df, on="lga_id", how="left")
    export_geo.to_file(GOLD / "gold_exports_geojson.geojson", driver="GeoJSON")
    return export_df, export_geo


def build_gold_manifest(export_df: pd.DataFrame) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(export_df)),
        "columns": list(export_df.columns),
        "model_version": export_df["model_version"].dropna().unique().tolist() if "model_version" in export_df.columns else [],
    }
    (GOLD / "gold_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    build_gold_lga_risk()
    build_gold_lga_explain()
    build_gold_state_rollup()
    export_df, _ = build_gold_exports()
    build_gold_manifest(export_df)
    print("Gold data build complete")


if __name__ == "__main__":
    main()
