"""Lightweight artifact checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "path",
    [
        Path("data/processed/lga_features.csv"),
        Path("data/processed/lga_predictions.csv"),
        Path("models/logreg.pkl"),
        Path("models/xgb.pkl"),
    ],
)
def test_artifacts_exist(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} missing; run pipeline first.")
    assert path.exists()


def test_feature_ranges() -> None:
    path = Path("data/processed/lga_features.csv")
    if not path.exists():
        pytest.skip("Features missing; run pipeline first.")
    df = pd.read_csv(path)
    assert (df["avg_distance_km"].between(0, 200)).all()
    assert (df["facilities_per_10k"].fillna(0) >= 0).all()


def test_prediction_ranges() -> None:
    path = Path("data/processed/lga_predictions.csv")
    if not path.exists():
        pytest.skip("Predictions missing; run pipeline first.")
    df = pd.read_csv(path)
    assert df["risk_prob"].between(0, 1).all()
