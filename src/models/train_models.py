"""Train versioned Nigeria risk models with a 2024 travel-time recalibration stage."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import DMatrix, XGBRegressor

matplotlib.use("Agg")

SEED = 42
MODEL_VERSION = "v1.4"
MODELS_DIR = Path("models")
VERSIONED_MODEL_DIR = MODELS_DIR / f"risk_model_{MODEL_VERSION}"
V13_MODEL_DIR = MODELS_DIR / "risk_model_v1.3"
STAGE1_FEATURES = [
    "facilities_per_10k",
    "avg_distance_km",
    "u5mr_mean",
    "coverage_5km",
    "towers_per_10k",
    "population_density",
]
STAGE2_FEATURES = ["pop_pct_60min"]
FEATURE_COLS = [*STAGE1_FEATURES, *STAGE2_FEATURES]
HOLDOUT_YEAR = 2024
TRAIN_YEARS = (2013, 2018)
STAGE2_BLEND_WEIGHT = 0.25
STAGE2_RIDGE_ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0]
BASE_TARGET_WEIGHTS = {
    "mortality": 0.28,
    "facility": 0.22,
    "distance": 0.18,
    "coverage5": 0.12,
    "connectivity": 0.10,
    "density": 0.10,
}


class TwoStageRiskModel:
    """Stage 1 base scorer plus a 2024-only routed-access adjustment."""

    def __init__(
        self,
        stage1_model: XGBRegressor,
        stage1_features: list[str],
        stage2_model: RidgeCV | None = None,
        stage2_features: list[str] | None = None,
        years_with_stage2: list[int] | None = None,
    ) -> None:
        self.stage1_model = stage1_model
        self.stage1_features = list(stage1_features)
        self.stage2_model = stage2_model
        self.stage2_features = list(stage2_features or [])
        self.years_with_stage2 = list(years_with_stage2 or [])

    @staticmethod
    def _frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def predict_stage1(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._frame(X)
        X_stage1 = frame[self.stage1_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        preds = self.stage1_model.predict(X_stage1)
        return np.clip(np.asarray(preds, dtype=float), 0.0, 10.0)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._frame(X)
        preds = self.predict_stage1(frame)
        if self.stage2_model is None or not self.stage2_features:
            return preds

        X_stage2 = frame[self.stage2_features].apply(pd.to_numeric, errors="coerce")
        mask = X_stage2.notna().all(axis=1)
        if mask.any():
            adjustments = self.stage2_model.predict(X_stage2.loc[mask])
            preds = preds.copy()
            preds[mask.to_numpy()] = preds[mask.to_numpy()] + np.asarray(adjustments, dtype=float)
        return np.clip(preds, 0.0, 10.0)


class ProbabilityToScoreModel:
    """Legacy wrapper kept for v1.3 artifact compatibility."""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(frame)[:, 1] * 10.0
        return self.estimator.predict(frame)


sys.modules.setdefault("src.models.train_models", sys.modules[__name__])
TwoStageRiskModel.__module__ = "src.models.train_models"
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


def _temporal_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    years = pd.to_numeric(df["year"], errors="coerce")
    train_mask = years.isin(TRAIN_YEARS)
    valid_mask = years == HOLDOUT_YEAR
    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Expected 2013+2018 rows for training and 2024 rows for holdout.")
    return train_mask, valid_mask


def _rank_severity(series: pd.Series, *, higher_is_worse: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    if valid.sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if valid.sum() == 1:
        base = pd.Series(5.0, index=series.index, dtype=float)
        return base.where(valid, np.nan)

    ranked = pd.Series(rankdata(numeric[valid], method="average"), index=numeric[valid].index, dtype=float)
    scaled = (ranked - 1.0) / (len(ranked) - 1.0) * 10.0
    if not higher_is_worse:
        scaled = 10.0 - scaled
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[scaled.index] = scaled
    return out


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year, frame in df.groupby("year"):
        block = pd.DataFrame(index=frame.index)
        block["target_mortality"] = _rank_severity(frame["u5mr_mean"], higher_is_worse=True)
        block["target_facility"] = _rank_severity(frame["facilities_per_10k"], higher_is_worse=False)
        block["target_distance"] = _rank_severity(frame["avg_distance_km"], higher_is_worse=True)
        block["target_coverage5"] = _rank_severity(frame["coverage_5km"], higher_is_worse=False)
        block["target_connectivity"] = _rank_severity(frame["towers_per_10k"], higher_is_worse=False)
        block["target_density"] = _rank_severity(frame["population_density"], higher_is_worse=True)
        block["target_access60"] = _rank_severity(frame["pop_pct_60min"], higher_is_worse=False)
        block["year"] = year
        parts.append(block)

    targets = pd.concat(parts).sort_index()
    target_cols = [
        "target_mortality",
        "target_facility",
        "target_distance",
        "target_coverage5",
        "target_connectivity",
        "target_density",
        "target_access60",
    ]
    df = pd.concat([df, targets[target_cols]], axis=1)

    base_target = (
        df["target_mortality"] * BASE_TARGET_WEIGHTS["mortality"]
        + df["target_facility"] * BASE_TARGET_WEIGHTS["facility"]
        + df["target_distance"] * BASE_TARGET_WEIGHTS["distance"]
        + df["target_coverage5"] * BASE_TARGET_WEIGHTS["coverage5"]
        + df["target_connectivity"] * BASE_TARGET_WEIGHTS["connectivity"]
        + df["target_density"] * BASE_TARGET_WEIGHTS["density"]
    )
    df["target_stage1"] = base_target.clip(0.0, 10.0)
    df["target_stage2"] = df["target_stage1"]

    routed_mask = df["pop_pct_60min"].notna()
    df.loc[routed_mask, "target_stage2"] = (
        df.loc[routed_mask, "target_stage1"] * (1.0 - STAGE2_BLEND_WEIGHT)
        + df.loc[routed_mask, "target_access60"] * STAGE2_BLEND_WEIGHT
    ).clip(0.0, 10.0)
    return df


def _build_stage1_search() -> RandomizedSearchCV:
    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_estimators=350,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    params = {
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "reg_lambda": [1.0, 1.5, 2.0],
    }
    return RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=10,
        cv=3,
        random_state=SEED,
        n_jobs=1,
    )


def _regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "spearman": float(spearmanr(y_true_arr, y_pred_arr).statistic),
    }


def _load_legacy_v13_scores(df_2024: pd.DataFrame) -> np.ndarray | None:
    model_path = V13_MODEL_DIR / "model.joblib"
    feature_path = V13_MODEL_DIR / "feature_importance.csv"
    if not model_path.exists() or not feature_path.exists():
        return None
    legacy_model = joblib.load(model_path)
    legacy_features = pd.read_csv(feature_path)["feature"].dropna().astype(str).tolist()
    X_legacy = df_2024[legacy_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    preds = legacy_model.predict(X_legacy)
    return np.clip(np.asarray(preds, dtype=float), 0.0, 10.0)


def _fit_stage2_oof(
    X_2024: pd.DataFrame,
    residual_2024: pd.Series,
    alphas: list[float],
) -> tuple[np.ndarray, RidgeCV]:
    X_reset = X_2024.reset_index(drop=True)
    y_reset = residual_2024.reset_index(drop=True)
    oof = np.zeros(len(X_reset), dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in splitter.split(X_reset):
        model = RidgeCV(alphas=alphas)
        model.fit(X_reset.iloc[train_idx], y_reset.iloc[train_idx])
        oof[valid_idx] = model.predict(X_reset.iloc[valid_idx])

    final_model = RidgeCV(alphas=alphas)
    final_model.fit(X_reset, y_reset)
    return oof, final_model


def _stage1_shap_values(model: XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    booster = model.get_booster()
    matrix = DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(matrix, pred_contribs=True)
    shap_df = pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)
    return shap_df


def _stage2_shap_values(model: RidgeCV, X: pd.Series, mean_x: float) -> pd.Series:
    coef = float(model.coef_[0]) if np.ndim(model.coef_) else float(model.coef_)
    return coef * (pd.to_numeric(X, errors="coerce").fillna(mean_x) - mean_x)


def _build_shap_outputs(
    df: pd.DataFrame,
    stage1_model: XGBRegressor,
    stage2_model: RidgeCV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_stage1 = df[STAGE1_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    stage1_shap = _stage1_shap_values(stage1_model, X_stage1)

    shap_df = pd.DataFrame(index=df.index)
    for col in STAGE1_FEATURES:
        shap_df[col] = stage1_shap[col]

    stage2_raw = pd.to_numeric(df["pop_pct_60min"], errors="coerce")
    stage2_mean = float(stage2_raw.dropna().mean()) if stage2_raw.notna().any() else 0.0
    shap_df["pop_pct_60min"] = _stage2_shap_values(stage2_model, stage2_raw, stage2_mean)
    shap_df.loc[stage2_raw.isna(), "pop_pct_60min"] = 0.0

    shap_df.insert(0, "lga_name", df["lga_name"].values)
    shap_df.insert(1, "year", pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int).values)
    shap_df.to_csv("data/processed/shap_values.csv", index=False)

    shap_2024 = shap_df[shap_df["year"] == HOLDOUT_YEAR]
    feature_importance = (
        shap_2024[FEATURE_COLS]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return shap_df, feature_importance


def _metadata(pop_coverage: dict[str, int]) -> dict[str, object]:
    return {
        "model_version": MODEL_VERSION,
        "strategy": "two_stage",
        "stage1_features": STAGE1_FEATURES,
        "stage2_features": STAGE2_FEATURES,
        "years_with_stage2": [HOLDOUT_YEAR],
        "pop_pct_60min_coverage": pop_coverage,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "holdout_year": HOLDOUT_YEAR,
        "train_years": list(TRAIN_YEARS),
        "evaluation_protocol": "stage1 temporal holdout on 2024; stage2 5-fold OOF recalibration on 2024 residuals",
        "target_notes": {
            "stage1": "Weighted per-year rank blend of mortality, facilities, distance, 5km coverage, connectivity, and density.",
            "stage2": "2024-only routed-access residual adjustment using pop_pct_60min.",
        },
    }


def _build_model_card(
    *,
    v13_metrics: dict[str, float] | None,
    v14_metrics: dict[str, float],
    shap_top5: pd.DataFrame,
) -> str:
    def _fmt(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.3f}"

    rows = [
        ("Spearman rho", _fmt(v13_metrics.get("spearman") if v13_metrics else None), _fmt(v14_metrics["spearman"])),
        ("MAE", _fmt(v13_metrics.get("mae") if v13_metrics else None), _fmt(v14_metrics["mae"])),
        ("RMSE", _fmt(v13_metrics.get("rmse") if v13_metrics else None), _fmt(v14_metrics["rmse"])),
        ("R2", _fmt(v13_metrics.get("r2") if v13_metrics else None), _fmt(v14_metrics["r2"])),
    ]
    table = "\n".join(f"| {metric} | {left} | {right} |" for metric, left, right in rows)
    top5_lines = "\n".join(f"- {row.feature}: {row.importance:.4f}" for row in shap_top5.itertuples(index=False))
    return "\n".join(
        [
            "# HDS Nigeria Risk Model v1.4",
            "",
            "## Summary",
            "Version 1.4 keeps the all-year base scorer on the shared access and mortality feature set, then adds a 2024-only routed travel-time recalibration so LGAs with weak road access move upward without inventing historical drive-time values for 2013 or 2018.",
            "",
            "## Training data",
            "- DHS 2013, 2018 (train)",
            "- DHS 2024 (temporal holdout)",
            "",
            "## Features",
            "### Stage 1 (all years)",
            *[f"- {feature}" for feature in STAGE1_FEATURES],
            "",
            "### Stage 2 - 2024 only",
            "- pop_pct_60min: % of LGA population within 60-min drive of any health facility",
            "  Source: OpenRouteService isochrones against NHFR 51,022 facilities (1km deduplicated to 35,732)",
            "  Coverage: 2024 only. Not available for 2013/2018.",
            "",
            "## Evaluation (2024 holdout)",
            "| Metric | v1.3 | v1.4 |",
            "|--------|------|------|",
            table,
            "",
            "## 2024 SHAP top 5",
            top5_lines,
            "",
            "## Known limitations",
            "- pop_pct_60min uses driving-car profile only. Walking/motorcycle access not modelled. May understate access barriers in areas where vehicles are uncommon.",
            "- OSM road network quality varies by region. Northwest Nigeria road data is less complete than Southwest.",
            "- 2013 and 2018 scores do not incorporate travel time. Cross-year comparisons should account for this methodological difference.",
            "",
            "## Deployment",
            "- Scores are rank-normalized within year to [0, 10]",
            "- Risk score > 5.5 = above national median (higher risk)",
            "- Risk score <= 5.5 = below national median (lower risk)",
            "",
        ]
    )


def _write_versioned_artifacts(
    *,
    model: TwoStageRiskModel,
    feature_importance: pd.DataFrame,
    metrics: dict[str, object],
    metadata: dict[str, object],
    model_card: str,
) -> None:
    VERSIONED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, VERSIONED_MODEL_DIR / "model.joblib")
    feature_importance.to_csv(VERSIONED_MODEL_DIR / "feature_importance.csv", index=False)
    feature_hash = hashlib.sha256("\n".join(feature_importance["feature"].tolist()).encode("utf-8")).hexdigest()
    (VERSIONED_MODEL_DIR / "feature_list_hash.txt").write_text(feature_hash, encoding="utf-8")
    (VERSIONED_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (VERSIONED_MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (VERSIONED_MODEL_DIR / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (VERSIONED_MODEL_DIR / "model_card.md").write_text(model_card, encoding="utf-8")


def train_models(features_path: Path) -> dict[str, object]:
    df = _build_targets(_load_features(features_path))
    year_col = pd.to_numeric(df["year"], errors="coerce")
    train_mask, valid_mask = _temporal_split(df)

    X_stage1 = df[STAGE1_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_stage1 = pd.to_numeric(df["target_stage1"], errors="coerce")

    search = _build_stage1_search()
    search.fit(X_stage1.loc[train_mask], y_stage1.loc[train_mask])
    stage1_model = search.best_estimator_

    stage1_pred_all = np.clip(stage1_model.predict(X_stage1), 0.0, 10.0)
    stage1_pred_2024 = stage1_pred_all[valid_mask.to_numpy()]
    X_stage2_2024 = df.loc[valid_mask, STAGE2_FEATURES].apply(pd.to_numeric, errors="coerce")
    target_2024 = pd.to_numeric(df.loc[valid_mask, "target_stage2"], errors="coerce")
    residual_2024 = target_2024 - stage1_pred_2024
    oof_adjustments, stage2_model = _fit_stage2_oof(X_stage2_2024, residual_2024, STAGE2_RIDGE_ALPHAS)
    v14_pred_2024 = np.clip(stage1_pred_2024 + oof_adjustments, 0.0, 10.0)

    final_model = TwoStageRiskModel(
        stage1_model=stage1_model,
        stage1_features=STAGE1_FEATURES,
        stage2_model=stage2_model,
        stage2_features=STAGE2_FEATURES,
        years_with_stage2=[HOLDOUT_YEAR],
    )

    full_scores = final_model.predict(df[FEATURE_COLS])
    predictions = pd.DataFrame(
        {
            "lga_name": df["lga_name"],
            "year": year_col.fillna(HOLDOUT_YEAR).astype(int),
            "risk_prob": np.clip(full_scores / 10.0, 0.0, 1.0),
            "risk_score_total": full_scores,
            "risk_label": (full_scores >= 5.5).astype(int),
            "fold": np.where(valid_mask, "holdout_2024", "train_2013_2018"),
            "model_version": MODEL_VERSION,
            "stage": np.where(df["pop_pct_60min"].notna(), "stage1_plus_stage2", "stage1_only"),
        }
    )
    pred_path = Path("data/processed/lga_predictions.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)

    shap_df, feature_importance = _build_shap_outputs(df, stage1_model, stage2_model)
    feature_importance.to_csv("docs/xgb_feature_importance.csv", index=False)

    v13_pred_2024 = _load_legacy_v13_scores(df.loc[valid_mask, STAGE1_FEATURES])
    v13_metrics = _regression_metrics(target_2024, v13_pred_2024) if v13_pred_2024 is not None else None
    v14_stage1_metrics = _regression_metrics(target_2024, stage1_pred_2024)
    v14_metrics = _regression_metrics(target_2024, v14_pred_2024)

    coverage = {
        str(int(year)): int(pd.to_numeric(group["pop_pct_60min"], errors="coerce").notna().sum())
        for year, group in df.groupby("year")
    }
    metadata = _metadata(coverage)

    shap_top5 = feature_importance.head(5).copy()
    metrics = {
        "model_version": MODEL_VERSION,
        "strategy": "two_stage",
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "holdout_year": HOLDOUT_YEAR,
        "v1_3_baseline": v13_metrics,
        "v1_4_stage1": v14_stage1_metrics,
        "v1_4": v14_metrics,
        "stage2_blend_weight": STAGE2_BLEND_WEIGHT,
        "stage2_ridge_alpha": float(stage2_model.alpha_),
        "stage2_ridge_coef": float(stage2_model.coef_[0]) if np.ndim(stage2_model.coef_) else float(stage2_model.coef_),
        "stage2_ridge_intercept": float(stage2_model.intercept_),
    }
    model_card = _build_model_card(v13_metrics=v13_metrics, v14_metrics=v14_metrics, shap_top5=shap_top5)
    _write_versioned_artifacts(
        model=final_model,
        feature_importance=feature_importance,
        metrics=metrics,
        metadata=metadata,
        model_card=model_card,
    )

    logging.info("OK: trained two-stage v1.4 model and generated outputs.")
    print(
        f"OK: trained models | version={MODEL_VERSION} | predictions rows={len(predictions)} | "
        f"holdout_rows={int(valid_mask.sum())}"
    )
    return {
        "metrics": metrics,
        "metadata": metadata,
        "feature_importance": feature_importance,
        "shap": shap_df,
    }


def main() -> None:
    _configure_logging()
    features_path = Path("data/processed/lga_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Missing features file. Run python -m src.data.build_features first.")
    train_models(features_path)


if __name__ == "__main__":
    main()
