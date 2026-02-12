"""Contract validation for app-ready gold outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_FILE = ROOT / "data" / "gold" / "gold_lga_risk.csv"


def main() -> None:
    df = pd.read_csv(GOLD_FILE)
    required = {
        "lga_id",
        "state_id",
        "year",
        "risk_score_total",
        "confidence_pct",
        "confidence_reason_codes",
        "estimate_as_of",
        "model_version",
    }
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")
    if not df["risk_score_total"].between(0, 10).all():
        raise AssertionError("risk_score_total must be between 0 and 10")
    if not df["confidence_pct"].between(0, 100).all():
        raise AssertionError("confidence_pct must be between 0 and 100")
    if df[["lga_id", "year"]].duplicated().any():
        raise AssertionError("lga_id/year combinations must be unique")
    if df["risk_score_total"].isna().any():
        raise AssertionError("risk_score_total cannot be null")
    print("Gold contracts validated")


if __name__ == "__main__":
    main()
