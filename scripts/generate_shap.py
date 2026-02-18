#!/usr/bin/env python3
"""Generate per-LGA feature attribution values for the research view.

Output: data/processed/shap_values.csv
Columns: lga_name, year, <feature columns...>

Fallback order:
1) Load a saved model artifact and compute attributions
2) Retrain from lga_features + lga_predictions and compute attributions
3) Synthetic z-score proxies (marked with is_synthetic=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    from xgboost import Booster, DMatrix
except Exception:  # pragma: no cover - optional dependency
    Booster = None
    DMatrix = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("generate_shap")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DATA_MODELS_DIR = ROOT / "data" / "models"
OUTPUT_PATH = PROCESSED_DIR / "shap_values.csv"

IDENTIFIER_COLS = {
    "lga_name",
    "state_name",
    "lga_id",
    "state_id",
    "year",
    "risk_score",
    "risk_prob",
    "risk_label",
    "target",
    "geometry",
    "lga_uid",
    "fold",
}
LEGACY_TRAIN_FEATURES = [
    "u5mr_mean",
    "u5mr_median",
    "facilities_per_10k",
    "avg_distance_km",
    "urban_prop",
    "population",
    "population_density",
    "coverage_5km",
]
RISK_ASCENDING_PROXY = {"facilities_per_10k", "towers_per_10k", "coverage_5km"}


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "lga_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")
    df = pd.read_csv(path)
    if "towers_per_10k_pop" in df.columns and "towers_per_10k" not in df.columns:
        df = df.rename(columns={"towers_per_10k_pop": "towers_per_10k"})
    if "lga_name" not in df.columns:
        raise ValueError("lga_features.csv must include lga_name")
    return df


def _infer_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in IDENTIFIER_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found in lga_features.csv")
    return cols


def _candidate_model_paths() -> list[Path]:
    priority = [
        MODELS_DIR / "xgb.pkl",
        MODELS_DIR / "risk_model_v1.2" / "model.joblib",
        MODELS_DIR / "logreg.pkl",
    ]
    candidates: list[Path] = []
    for path in priority:
        if path.exists():
            candidates.append(path)

    patterns = ["*.joblib", "*.pkl", "*.json", "*.ubj", "*.h5", "*.pt"]
    for root in (MODELS_DIR, DATA_MODELS_DIR):
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(root.rglob(pattern)))

    # Keep order stable while removing duplicates.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _unwrap_loaded_model(model_obj: Any) -> tuple[Any, list[str] | None]:
    if isinstance(model_obj, dict) and "model" in model_obj:
        hinted = model_obj.get("feature_cols")
        hinted_cols = [str(c) for c in hinted] if isinstance(hinted, (list, tuple)) else None
        return model_obj["model"], hinted_cols
    return model_obj, None


def load_saved_model() -> tuple[Any, Path, list[str] | None]:
    for path in _candidate_model_paths():
        suffix = path.suffix.lower()
        try:
            if suffix in {".joblib", ".pkl"}:
                loaded = joblib.load(path)
                model, hinted_cols = _unwrap_loaded_model(loaded)
                LOG.info("Loaded model artifact: %s", path)
                return model, path, hinted_cols
            if suffix in {".json", ".ubj"} and Booster is not None:
                booster = Booster()
                booster.load_model(str(path))
                LOG.info("Loaded XGBoost booster artifact: %s", path)
                return booster, path, None
            # .h5/.pt are acknowledged but unsupported in this code path.
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("Skipping unreadable model %s (%s)", path, exc)
            continue

    raise FileNotFoundError(
        "No readable saved model artifact found under models/ or data/models/"
    )


def _feature_cols_from_sidecar(model_path: Path) -> list[str] | None:
    candidates = [
        model_path.with_name("feature_importance.csv"),
        model_path.parent / "feature_importance.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        sidecar = pd.read_csv(path)
        if "feature" not in sidecar.columns:
            continue
        cols = sidecar["feature"].dropna().astype(str).tolist()
        if cols:
            return cols
    return None


def resolve_feature_cols(
    df: pd.DataFrame,
    model: Any,
    model_path: Path,
    hinted_cols: list[str] | None = None,
) -> list[str]:
    cols: list[str] = []
    if hinted_cols:
        cols = [str(c) for c in hinted_cols]
    elif hasattr(model, "feature_names_in_"):
        cols = [str(c) for c in getattr(model, "feature_names_in_")]
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        if booster.feature_names:
            cols = [str(c) for c in booster.feature_names]
    elif Booster is not None and isinstance(model, Booster):
        if model.feature_names:
            cols = [str(c) for c in model.feature_names]

    if not cols:
        sidecar_cols = _feature_cols_from_sidecar(model_path)
        if sidecar_cols:
            cols = sidecar_cols

    if not cols and model_path.name.lower() == "xgb.pkl":
        cols = [c for c in LEGACY_TRAIN_FEATURES if c in df.columns]

    if not cols:
        cols = _infer_numeric_feature_cols(df)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Model expects missing feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return cols


def _coerce_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0)
    return X


def _normalize_shap_output(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values"):
        values = values.values

    arr = np.asarray(values)
    if arr.ndim == 3:
        # Common in newer SHAP APIs for binary/multiclass outputs.
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    return arr


def _xgb_pred_contribs(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if DMatrix is None:
        return None

    booster = None
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif Booster is not None and isinstance(model, Booster):
        booster = model
    if booster is None:
        return None

    dmat = DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    contribs = np.asarray(contribs)
    if contribs.ndim != 2:
        return None
    if contribs.shape[1] == X.shape[1] + 1:
        contribs = contribs[:, :-1]  # Drop bias term.
    return contribs


def compute_attributions(df: pd.DataFrame, model: Any, feature_cols: list[str]) -> np.ndarray:
    X = _coerce_feature_matrix(df, feature_cols)
    LOG.info("Computing attributions for %d rows x %d features", len(X), len(feature_cols))

    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            return _normalize_shap_output(explainer.shap_values(X))
        except Exception as exc:
            LOG.warning("TreeExplainer failed (%s); trying KernelExplainer", exc)

        try:
            if not hasattr(model, "predict_proba"):
                raise TypeError("Model has no predict_proba for KernelExplainer fallback")
            background = shap.sample(X, min(100, len(X)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            return _normalize_shap_output(explainer.shap_values(X, nsamples=100))
        except Exception as exc:
            LOG.warning("KernelExplainer failed (%s); trying model-native contributions", exc)
    else:
        LOG.warning("shap package is not installed; trying model-native contributions")

    contribs = _xgb_pred_contribs(model, X)
    if contribs is not None:
        return contribs

    raise RuntimeError("Could not compute SHAP values from available explainers")


def retrain_fallback_model(features: pd.DataFrame) -> tuple[Any, list[str], Path]:
    preds_path = PROCESSED_DIR / "lga_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")
    preds = pd.read_csv(preds_path)
    if "risk_prob" not in preds.columns and "risk_score" in preds.columns:
        preds = preds.rename(columns={"risk_score": "risk_prob"})
    if "risk_prob" not in preds.columns and "risk_label" not in preds.columns:
        raise ValueError("lga_predictions.csv requires risk_prob or risk_label")

    join_keys = ["lga_name"]
    if "year" in features.columns and "year" in preds.columns:
        join_keys.append("year")

    pred_cols = join_keys + [c for c in ["risk_prob", "risk_label"] if c in preds.columns]
    train_df = features.merge(preds[pred_cols], on=join_keys, how="inner")
    if train_df.empty:
        raise ValueError("No rows matched between lga_features and lga_predictions for retraining")

    feature_cols = [c for c in LEGACY_TRAIN_FEATURES if c in train_df.columns]
    if len(feature_cols) < 3:
        feature_cols = _infer_numeric_feature_cols(train_df)

    X = _coerce_feature_matrix(train_df, feature_cols)
    if "risk_label" in train_df.columns:
        y = pd.to_numeric(train_df["risk_label"], errors="coerce")
    else:
        y = (pd.to_numeric(train_df["risk_prob"], errors="coerce") >= 0.5).astype(float)
    y = y.dropna().astype(int)
    X = X.loc[y.index]

    if y.nunique() < 2:
        raise ValueError("Retraining target has fewer than 2 classes")

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    DATA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_MODELS_DIR / "lga_risk_model.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, save_path)
    LOG.info("Saved fallback model to %s", save_path)
    return model, feature_cols, save_path


def build_output(
    df: pd.DataFrame,
    values: np.ndarray,
    feature_cols: list[str],
    synthetic: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame(values, columns=feature_cols, index=df.index)
    out.insert(0, "lga_name", df["lga_name"].values)
    if "year" in df.columns:
        out.insert(1, "year", df["year"].values)
    if synthetic:
        out["is_synthetic"] = True
    return out


def generate_synthetic_shap(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    LOG.warning("No usable model/explainer found; generating synthetic z-score proxies")
    X = _coerce_feature_matrix(df, feature_cols)
    std = X.std(ddof=0).replace(0, np.nan)
    zscores = ((X - X.mean()) / std).fillna(0.0)

    for col in RISK_ASCENDING_PROXY:
        if col in zscores.columns:
            zscores[col] = -zscores[col]

    return build_output(df, zscores.to_numpy(), feature_cols, synthetic=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    model_source = "unknown"

    try:
        model, model_path, hinted_cols = load_saved_model()
        feature_cols = resolve_feature_cols(features, model, model_path, hinted_cols)
        attributions = compute_attributions(features, model, feature_cols)
        shap_df = build_output(features, attributions, feature_cols, synthetic=False)
        model_source = f"saved:{model_path}"
    except Exception as saved_exc:
        LOG.warning("Saved-model path failed: %s", saved_exc)
        try:
            model, feature_cols, retrained_path = retrain_fallback_model(features)
            attributions = compute_attributions(features, model, feature_cols)
            shap_df = build_output(features, attributions, feature_cols, synthetic=False)
            model_source = f"retrained:{retrained_path}"
        except Exception as retrain_exc:
            LOG.warning("Retrain path failed: %s", retrain_exc)
            feature_cols = _infer_numeric_feature_cols(features)
            shap_df = generate_synthetic_shap(features, feature_cols)
            model_source = "synthetic_zscore"

    shap_df.to_csv(OUTPUT_PATH, index=False)
    LOG.info("Wrote %s (%d rows) using source=%s", OUTPUT_PATH, len(shap_df), model_source)

    value_cols = [c for c in shap_df.columns if c not in {"lga_name", "year", "is_synthetic"}]
    if value_cols:
        means = (
            shap_df[value_cols]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .mean()
            .sort_values(ascending=False)
            .head(8)
        )
        LOG.info("Top features by mean |attribution|:\n%s", means.to_string())


if __name__ == "__main__":
    main()

