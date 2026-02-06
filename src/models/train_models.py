"""Train risk models for Nigeria health desert scoring."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

SEED = 42


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
    return df


def _make_folds(df: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    coords = df[["lga_lat", "lga_lon"]].fillna(0.0)
    kmeans = KMeans(n_clusters=n_folds, random_state=SEED, n_init=10)
    return pd.Series(kmeans.fit_predict(coords), index=df.index, name="fold")


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
        n_jobs=-1,
    )
    return logreg, search


def _plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.savefig(out_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"PR AUC={average_precision_score(y_true, y_prob):.3f}")
    plt.savefig(out_dir / "pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def train_models(features_path: Path) -> None:
    df = _load_features(features_path)
    year_col = df["year"] if "year" in df.columns else None
    df["high_risk"] = df["u5mr_mean"] > df["u5mr_mean"].median()
    if df["high_risk"].nunique() < 2:
        raise ValueError("Need at least two classes for training.")

    feature_cols = [
        "u5mr_mean",
        "u5mr_median",
        "facilities_per_10k",
        "avg_distance_km",
        "urban_prop",
        "population",
        "population_density",
        "coverage_5km",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_cols].fillna(0.0)
    y = df["high_risk"].astype(int).values

    folds = _make_folds(df)
    df["fold"] = folds

    logreg, xgb_search = _build_models(feature_cols)
    logreg.fit(X, y)
    xgb_search.fit(X, y)

    joblib.dump(logreg, Path("models/logreg.pkl"))
    joblib.dump(xgb_search.best_estimator_, Path("models/xgb.pkl"))

    risk_prob = np.zeros(len(df))
    for fold in sorted(folds.unique()):
        train_idx = folds != fold
        test_idx = folds == fold
        model = xgb_search.best_estimator_
        model.fit(X[train_idx], y[train_idx])
        risk_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    if not np.all((risk_prob >= 0) & (risk_prob <= 1)):
        raise ValueError("risk_prob outside [0,1].")

    predictions = pd.DataFrame(
        {
            "lga_name": df["lga_name"],
            "year": year_col if year_col is not None else 2018,
            "risk_prob": risk_prob,
            "risk_label": (risk_prob >= 0.5).astype(int),
            "fold": folds,
        }
    )
    pred_path = Path("data/processed/lga_predictions.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)

    _plot_curves(y, risk_prob, Path("docs"))

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": xgb_search.best_estimator_.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv("docs/xgb_feature_importance.csv", index=False)

    if importlib.util.find_spec("shap") is None:
        logging.info("SHAP not available; skipping SHAP outputs.")
    else:
        import shap

        explainer = shap.TreeExplainer(xgb_search.best_estimator_)
        shap_values = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=feature_cols)
        shap_df.insert(0, "lga_name", df["lga_name"].values)
        if year_col is not None:
            shap_df.insert(1, "year", year_col.values)
        shap_df.to_csv("data/processed/shap_values.csv", index=False)

        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.savefig("docs/shap_importance.png", dpi=200, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("docs/shap_global.png", dpi=200, bbox_inches="tight")
        plt.close()

    logging.info("OK: trained models and generated outputs.")
    print(f"OK: trained models | predictions rows={len(predictions)}")


def main() -> None:
    _configure_logging()
    features_path = Path("data/processed/lga_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Missing features file. Run python -m src.data.build_features first.")
    train_models(features_path)


if __name__ == "__main__":
    main()
