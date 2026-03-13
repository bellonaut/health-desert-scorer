"""Train risk models for Nigeria health desert scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
MODEL_VERSION = "v1.3"
MODELS_DIR = Path("models")
VERSIONED_MODEL_DIR = MODELS_DIR / f"risk_model_{MODEL_VERSION}"
FEATURE_COLS = [
    "facilities_per_10k",
    "avg_distance_km",
    "u5mr_mean",
    "coverage_5km",
    "towers_per_10k",
    "population_density",
]


class ProbabilityToScoreModel:
    """Wrap a probabilistic classifier so `predict()` returns 0-10 risk scores."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(frame)[:, 1] * 10.0
        return self.estimator.predict(frame)


sys.modules.setdefault("src.models.train_models", sys.modules[__name__])
ProbabilityToScoreModel.__module__ = "src.models.train_models"


def _configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/train_models.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Features file is empty.")
    df = df.rename(columns={"towers_per_10k_pop": "towers_per_10k"})
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _build_models(feature_cols: list[str]) -> tuple[Pipeline, RandomizedSearchCV]:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), feature_cols)],
        remainder="drop",
    )
    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)),
        ]
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_params = {
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_lambda": [1.0, 1.5, 2.0],
    }
    search = RandomizedSearchCV(
        xgb,
        param_distributions=xgb_params,
        n_iter=10,
        cv=3,
        random_state=SEED,
        n_jobs=1,
    )
    return logreg, search


def _temporal_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "year" not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(False, index=df.index)

    years = pd.to_numeric(df["year"], errors="coerce")
    train_mask = years.isin([2013, 2018])
    valid_mask = years == 2024
    if train_mask.any() and valid_mask.any():
        return train_mask, valid_mask
    return pd.Series(True, index=df.index), pd.Series(False, index=df.index)


def _plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    if len(np.unique(y_true)) < 2:
        logging.warning("Validation labels contain a single class; skipping ROC/PR plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"PR AUC={average_precision_score(y_true, y_prob):.3f}")
    plt.savefig(out_dir / "pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def _evaluate_holdout(y_true: np.ndarray, y_prob: np.ndarray, *, train_rows: int, valid_rows: int) -> dict[str, float | int | str]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, float | int | str] = {
        "model_version": MODEL_VERSION,
        "train_years": "2013+2018",
        "validation_year": 2024,
        "train_rows": int(train_rows),
        "validation_rows": int(valid_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _write_versioned_artifacts(model, importance: pd.DataFrame, metrics: dict[str, float | int | str]) -> None:
    VERSIONED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ProbabilityToScoreModel(model), VERSIONED_MODEL_DIR / "model.joblib")
    importance.to_csv(VERSIONED_MODEL_DIR / "feature_importance.csv", index=False)
    feature_hash = hashlib.sha256("\n".join(importance["feature"].tolist()).encode("utf-8")).hexdigest()
    (VERSIONED_MODEL_DIR / "feature_list_hash.txt").write_text(feature_hash, encoding="utf-8")
    (VERSIONED_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (VERSIONED_MODEL_DIR / "model_card.md").write_text(
        "\n".join(
            [
                f"# Risk Model {MODEL_VERSION}",
                "",
                "## Intended Use",
                "Planning tool for identifying LGAs with healthcare access barriers in Nigeria.",
                "",
                "## Training Data",
                "- DHS Survey: 2013, 2018, 2024",
                "- Facilities: NHFR 2020",
                "- Population: WorldPop-derived LGA totals",
                "- Connectivity: OpenCellID",
                "",
                "## Validation",
                "- Train split: 2013 + 2018",
                "- Temporal holdout: 2024",
                "",
                "## Model Architecture",
                "- Algorithm: Gradient boosted tree classifier wrapped to emit 0-10 risk scores",
                f"- Features: {len(FEATURE_COLS)} core LGA-level inputs",
                "",
                "## Known Limitations",
                "- Does not capture seasonal road access shocks",
                "- Registry completeness still varies by state",
                "- Scores remain planning aids and require local validation",
                "",
            ]
        ),
        encoding="utf-8",
    )


def train_models(features_path: Path) -> None:
    df = _load_features(features_path)
    year_col = pd.to_numeric(df["year"], errors="coerce") if "year" in df.columns else pd.Series(np.nan, index=df.index)

    train_mask, valid_mask = _temporal_split(df)
    threshold = pd.to_numeric(df.loc[train_mask, "u5mr_mean"], errors="coerce").median()
    if pd.isna(threshold):
        threshold = pd.to_numeric(df["u5mr_mean"], errors="coerce").median()
    df["high_risk"] = pd.to_numeric(df["u5mr_mean"], errors="coerce") > float(threshold)
    if df["high_risk"].nunique() < 2:
        raise ValueError("Need at least two classes for training.")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["high_risk"].astype(int).values

    logreg, xgb_search = _build_models(FEATURE_COLS)
    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both classes.")

    logreg.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)
    holdout_model = xgb_search.best_estimator_

    if valid_mask.any():
        X_valid = X.loc[valid_mask]
        y_valid = y[valid_mask]
        valid_prob = holdout_model.predict_proba(X_valid)[:, 1]
        holdout_metrics = _evaluate_holdout(
            y_valid,
            valid_prob,
            train_rows=int(train_mask.sum()),
            valid_rows=int(valid_mask.sum()),
        )
        _plot_curves(y_valid, valid_prob, Path("docs"))
    else:
        holdout_metrics = {
            "model_version": MODEL_VERSION,
            "train_rows": int(train_mask.sum()),
            "validation_rows": 0,
            "note": "Temporal holdout unavailable; trained on all available rows.",
        }

    logreg_final, xgb_search_final = _build_models(FEATURE_COLS)
    logreg_final.fit(X, y)
    xgb_search_final.fit(X, y)
    final_xgb = xgb_search_final.best_estimator_

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg_final, MODELS_DIR / "logreg.pkl")
    joblib.dump(final_xgb, MODELS_DIR / "xgb.pkl")

    risk_prob = final_xgb.predict_proba(X)[:, 1]
    if not np.all((risk_prob >= 0) & (risk_prob <= 1)):
        raise ValueError("risk_prob outside [0,1].")

    predictions = pd.DataFrame(
        {
            "lga_name": df["lga_name"],
            "year": year_col.fillna(2024 if valid_mask.any() else 2018).astype(int),
            "risk_prob": risk_prob,
            "risk_label": (risk_prob >= 0.5).astype(int),
            "fold": np.where(valid_mask, "holdout_2024", "train_2013_2018"),
        }
    )
    pred_path = Path("data/processed/lga_predictions.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": final_xgb.feature_importances_,
        }
    )
    importance.sort_values("importance", ascending=False).to_csv("docs/xgb_feature_importance.csv", index=False)
    _write_versioned_artifacts(final_xgb, importance, holdout_metrics)

    if importlib.util.find_spec("shap") is None:
        logging.info("SHAP not available; skipping SHAP outputs.")
    else:
        import shap

        explainer = shap.TreeExplainer(final_xgb)
        shap_values = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
        shap_df.insert(0, "lga_name", df["lga_name"].values)
        if "year" in df.columns:
            shap_df.insert(1, "year", year_col.fillna(0).astype(int).values)
        shap_df.to_csv("data/processed/shap_values.csv", index=False)

        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.savefig("docs/shap_importance.png", dpi=200, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("docs/shap_global.png", dpi=200, bbox_inches="tight")
        plt.close()

    logging.info("OK: trained models and generated outputs.")
    print(
        f"OK: trained models | version={MODEL_VERSION} | predictions rows={len(predictions)} | "
        f"holdout_rows={int(valid_mask.sum())}"
    )


def main() -> None:
    _configure_logging()
    features_path = Path("data/processed/lga_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Missing features file. Run python -m src.data.build_features first.")
    train_models(features_path)


if __name__ == "__main__":
    main()
