"""Build gold-layer, app-ready decision tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models.score import score_lga

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"


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
    out["risk_score_mortality"] = (df["u5mr_mean"].fillna(0) / 20).clip(0, 10)
    return out


def build_gold_lga_risk() -> pd.DataFrame:
    GOLD.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SILVER / "lga_metrics.csv")
    quality = pd.read_csv(SILVER / "data_quality_profiles.csv")

    df = metrics.merge(quality, on=["lga_id", "year"], how="left")

    feature_cols = [
        "facilities_per_10k",
        "avg_distance_km",
        "u5mr_mean",
        "coverage_5km",
        "towers_per_10k",
        "population_density",
    ]

    try:
        model_scores = score_lga(df[feature_cols], version="v1.2")
        df["risk_score_total"] = pd.to_numeric(model_scores["risk_score_total"], errors="coerce")
        df["model_version"] = model_scores["model_version"]
    except Exception:
        # Deterministic fallback preserves reproducibility even without a local binary artifact.
        df["risk_score_total"] = (
            (10 - df["facilities_per_10k"].fillna(0).clip(0, 10)) * 0.35
            + (df["avg_distance_km"].fillna(0).clip(0, 20) / 2) * 0.20
            + (df["u5mr_mean"].fillna(0).clip(0, 200) / 20) * 0.25
            + (10 - (df["coverage_5km"].fillna(0).clip(0, 100) / 10)) * 0.20
        )
        df["model_version"] = "v1.2-fallback"

    components = _component_scores(df)
    df = pd.concat([df, components], axis=1)
    df["risk_score_total"] = df["risk_score_total"].clip(0, 10)
    df["confidence_reason_codes"] = df["confidence_issues"].fillna("none")
    df["confidence_pct"] = df["confidence_reason_codes"].apply(calculate_confidence)
    df["estimate_as_of"] = datetime.utcnow().strftime("%Y-%m-%d")

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


def main() -> None:
    build_gold_lga_risk()
    build_gold_lga_explain()
    print("Gold data build complete")


if __name__ == "__main__":
    main()
