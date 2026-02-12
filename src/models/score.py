"""Versioned scoring model loader and inference helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


MODELS_ROOT = Path("models")


def load_model(version: str = "v1.2"):
    """Load a versioned model artifact from models/risk_model_<version>/model.joblib."""
    if joblib is None:
        raise ImportError("joblib is required to load model artifacts")
    model_path = MODELS_ROOT / f"risk_model_{version}" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def get_required_features(version: str = "v1.2") -> list[str]:
    """Return the ordered feature list expected by model version."""
    feature_path = MODELS_ROOT / f"risk_model_{version}" / "feature_importance.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_path}")
    feature_df = pd.read_csv(feature_path)
    if "feature" not in feature_df.columns:
        raise ValueError("feature_importance.csv must include a 'feature' column")
    return feature_df["feature"].dropna().astype(str).tolist()


def score_lga(features: pd.DataFrame, version: str = "v1.2") -> pd.DataFrame:
    """Score LGAs and return model metadata with predictions."""
    required_features = get_required_features(version)
    missing = sorted(set(required_features) - set(features.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model = load_model(version)
    X = features[required_features].fillna(0)
    scores = model.predict(X)

    return pd.DataFrame(
        {
            "risk_score_total": scores,
            "model_version": version,
        },
        index=features.index,
    )
