"""Data loading and JSON-friendly helpers for the embedded Health Desert app."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

# Paths relative to repository root
ROOT_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = ROOT_DIR / "data" / "bronze"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SILVER_DIR = ROOT_DIR / "data" / "silver"
GOLD_DIR = ROOT_DIR / "data" / "gold"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Frontend focus labels -> dataframe columns
FOCUS_COLUMN = {
    "All risk": "risk_score",
    "Child mortality": "u5mr_mean",
    "Facility access": "facilities_per_10k",
    "Connectivity": "towers_per_10k",
    "5km coverage": "coverage_5km",
}

ASCENDING_FOCUS = {"facilities_per_10k", "towers_per_10k", "coverage_5km"}

# Optional softening for over-confident probabilities; env var lets us tune without code changes
RISK_TEMPERATURE = float(os.getenv("RISK_TEMPERATURE", "2.0"))

# Columns we expect to exist to avoid KeyErrors downstream
EXPECTED_COLUMNS = [
    "risk_score",
    "risk_score_total",
    "risk_score_facility_access",
    "risk_score_connectivity",
    "risk_score_mortality",
    "facilities_per_10k",
    "avg_distance_km",
    "u5mr_mean",
    "population",
    "population_density",
    "towers_per_10k",
    "coverage_5km",
    "shap_importance",
    "year",
    "confidence_pct",
    "confidence_reason_codes",
    "model_version",
    "estimate_as_of",
    "primary_barriers",
    "recommendation",
]


def _slug(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)


def _make_lga_id(lga_name: pd.Series, state_name: pd.Series | None = None) -> pd.Series:
    lga_slug = _slug(lga_name)
    if state_name is None:
        return lga_slug
    state_slug = _slug(state_name)
    state_slug = state_slug.where(state_name.notna() & state_name.astype(str).str.strip().ne(""), "")
    return (state_slug + "__" + lga_slug).where(state_slug.ne(""), lga_slug)


def _prefer_non_null(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Fill nulls in primary from fallback without relying on combine_first internals."""
    return primary.where(primary.notna(), fallback)


def _load_gold_risk() -> pd.DataFrame:
    gold_path = GOLD_DIR / "gold_lga_risk.csv"
    if not gold_path.exists():
        return pd.DataFrame()
    gold = pd.read_csv(gold_path)
    if "risk_score_total" in gold.columns:
        gold["risk_score"] = pd.to_numeric(gold["risk_score_total"], errors="coerce") / 10.0
    return gold


def _load_gold_explain() -> pd.DataFrame:
    path = GOLD_DIR / "gold_lga_explain.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_boundaries(resolution: str = "auto", is_mobile: bool | None = None, zoom: float | None = None) -> gpd.GeoDataFrame:
    """Load boundaries at the requested resolution with fallback to raw."""
    chosen = resolution
    if resolution == "auto":
        if is_mobile:
            chosen = "low"
        elif zoom is not None and zoom < 8:
            chosen = "medium"
        else:
            chosen = "high"

    file_map = {
        "low": SILVER_DIR / "lga_boundaries_lo.geojson",
        "medium": SILVER_DIR / "lga_boundaries_med.geojson",
        "high": SILVER_DIR / "lga_boundaries_hi.geojson",
    }
    path = file_map.get(chosen)
    if path is not None and path.exists():
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(RAW_DIR / "lga_boundaries.geojson")
        chosen = "raw"

    gdf = gdf.rename(columns={"lganame": "lga_name", "statename": "state_name", "uniq_id": "lga_uid"})
    gdf["lga_name"] = gdf["lga_name"].astype(str).str.strip()
    gdf["state_name"] = gdf["state_name"].astype(str).str.strip()
    if "lga_uid" not in gdf.columns:
        gdf["lga_uid"] = gdf["lga_name"].astype("category").cat.codes
    gdf["lga_uid"] = gdf["lga_uid"].astype(str)
    gdf["lga_id"] = _make_lga_id(gdf["lga_name"], gdf["state_name"])
    gdf["state_id"] = _slug(gdf["state_name"])
    gdf.attrs["boundary_resolution"] = chosen
    return gdf


def _load_features() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "lga_features.csv")
    features = features.rename(columns={"towers_per_10k_pop": "towers_per_10k"})
    if "towers_per_10k" not in features.columns and {"towers_count", "population"} <= set(features.columns):
        features["towers_per_10k"] = (
            pd.to_numeric(features["towers_count"], errors="coerce")
            / pd.to_numeric(features["population"], errors="coerce")
            * 10000
        )
    features["lga_id"] = _make_lga_id(features["lga_name"], features["state_name"])
    features["state_id"] = _slug(features["state_name"])
    return features


def _load_predictions() -> pd.DataFrame:
    preds_path = PROCESSED_DIR / "lga_predictions.csv"
    if not preds_path.exists():
        return pd.DataFrame()
    preds = pd.read_csv(preds_path)
    if "risk_score" not in preds.columns and "risk_prob" in preds.columns:
        preds = preds.rename(columns={"risk_prob": "risk_score"})
    if {"lga_name", "year", "risk_score"} <= set(preds.columns):
        preds = preds.groupby(["lga_name", "year"], as_index=False)["risk_score"].mean()
    return preds


def _temperature_scale(probs: pd.Series, temperature: float = 1.0) -> pd.Series:
    """Soften extreme probabilities by dividing logits by a temperature > 1.

    Keeps values in (0,1) and leaves NaNs untouched.
    """
    if temperature <= 0:
        return probs
    eps = 1e-6
    clipped = pd.to_numeric(probs, errors="coerce").clip(eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    scaled = logits / temperature
    softened = 1 / (1 + np.exp(-scaled))
    return softened


def _load_shap() -> pd.DataFrame | None:
    shap_path = PROCESSED_DIR / "shap_values.csv"
    if not shap_path.exists():
        return None
    return pd.read_csv(shap_path)


def _data_last_updated(gold_df: pd.DataFrame) -> str | None:
    if gold_df.empty or "estimate_as_of" not in gold_df.columns:
        return None
    return str(pd.to_datetime(gold_df["estimate_as_of"], errors="coerce").max().date())


@lru_cache(maxsize=4)
def load_backend_data(
    source_mode: str = "gold_first",
    boundary_resolution: str = "auto",
    is_mobile: bool | None = None,
    zoom: float | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None]:
    """Load all source data once per process."""
    boundaries = load_boundaries(resolution=boundary_resolution, is_mobile=is_mobile, zoom=zoom)
    features = _load_features()
    preds = _load_predictions()
    gold_risk = _load_gold_risk()
    gold_explain = _load_gold_explain()
    shap_df = _load_shap()

    merged_features = features.merge(preds, on=["lga_name", "year"], how="left")
    data_source_mode = "processed_fallback"

    if source_mode != "processed_only" and not gold_risk.empty:
        gold_risk = gold_risk.copy()
        if "lga_id" not in gold_risk.columns and "lga_name" in gold_risk.columns:
            gold_risk["lga_id"] = _make_lga_id(gold_risk["lga_name"], gold_risk.get("state_name"))
        gold_cols = [
            c
            for c in [
                "lga_id",
                "year",
                "risk_score",
                "risk_score_total",
                "risk_score_facility_access",
                "risk_score_connectivity",
                "risk_score_mortality",
                "confidence_pct",
                "confidence_reason_codes",
                "model_version",
                "estimate_as_of",
                "primary_barriers",
                "recommendation",
            ]
            if c in gold_risk.columns
        ]
        merged_features = merged_features.drop(columns=[c for c in ["risk_score", "risk_prob"] if c in merged_features.columns])
        merged_features = merged_features.merge(gold_risk[gold_cols], on=["lga_id", "year"], how="left")
        data_source_mode = "gold_first"

        if not gold_explain.empty:
            explain_cols = [c for c in ["lga_id", "year", "primary_barriers", "recommendation"] if c in gold_explain.columns]
            if explain_cols:
                merged_features = merged_features.drop(columns=[c for c in ["primary_barriers", "recommendation"] if c in merged_features.columns])
                merged_features = merged_features.merge(gold_explain[explain_cols], on=["lga_id", "year"], how="left")
    else:
        LOGGER.warning("Gold tables not found. Falling back to processed predictions for one release.")

    if "risk_score" not in merged_features.columns:
        merged_features["risk_score"] = pd.to_numeric(merged_features.get("risk_prob"), errors="coerce")
    merged_features["risk_score"] = pd.to_numeric(merged_features["risk_score"], errors="coerce").clip(0, 1)
    if RISK_TEMPERATURE and RISK_TEMPERATURE != 1.0:
        merged_features["risk_score"] = _temperature_scale(merged_features["risk_score"], RISK_TEMPERATURE)

    merged = boundaries.merge(merged_features, on=["lga_id"], how="left")
    if "lga_uid_x" in merged.columns or "lga_uid_y" in merged.columns:
        left = merged["lga_uid_y"] if "lga_uid_y" in merged.columns else None
        right = merged["lga_uid_x"] if "lga_uid_x" in merged.columns else None
        if left is None:
            merged["lga_uid"] = right
        elif right is None:
            merged["lga_uid"] = left
        else:
            merged["lga_uid"] = _prefer_non_null(left, right)
        merged = merged.drop(columns=[col for col in ["lga_uid_x", "lga_uid_y"] if col in merged.columns])
    if "lga_uid" not in merged.columns:
        merged["lga_uid"] = merged["lga_name"].astype("category").cat.codes

    # Resolve potential duplicate LGA columns after merge
    if "lga_name_x" in merged.columns or "lga_name_y" in merged.columns:
        left = merged["lga_name_y"] if "lga_name_y" in merged.columns else None
        right = merged["lga_name_x"] if "lga_name_x" in merged.columns else None
        if left is None:
            merged["lga_name"] = right
        elif right is None:
            merged["lga_name"] = left
        else:
            merged["lga_name"] = _prefer_non_null(left, right)
        merged = merged.drop(columns=[col for col in ["lga_name_x", "lga_name_y"] if col in merged.columns])

    # Resolve potential duplicate state columns after merge
    if "state_name_x" in merged.columns or "state_name_y" in merged.columns:
        left = merged["state_name_y"] if "state_name_y" in merged.columns else None
        right = merged["state_name_x"] if "state_name_x" in merged.columns else None
        if left is None:
            merged["state_name"] = right
        elif right is None:
            merged["state_name"] = left
        else:
            merged["state_name"] = _prefer_non_null(left, right)
        merged = merged.drop(columns=[col for col in ["state_name_x", "state_name_y"] if col in merged.columns])

    if shap_df is not None:
        shap_df = shap_df.rename(columns={"lganame": "lga_name"})
        feature_cols = [c for c in shap_df.columns if c not in {"lga_name", "year"}]
        if feature_cols:
            shap_df["shap_importance"] = shap_df[feature_cols].abs().sum(axis=1)
            join_keys = ["lga_name", "year"] if "year" in shap_df.columns else ["lga_name"]
            merged = merged.merge(
                shap_df[join_keys + ["shap_importance"]] if "shap_importance" in shap_df.columns else shap_df,
                on=join_keys,
                how="left",
            )
        else:
            merged["shap_importance"] = np.nan
    else:
        merged["shap_importance"] = np.nan

    for col in EXPECTED_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["lga_uid"] = merged["lga_uid"].astype(str)
    merged["state_name"] = merged["state_name"].astype(str).str.strip()
    merged["lga_name"] = merged["lga_name"].astype(str).str.strip()

    merged = gpd.GeoDataFrame(merged)
    merged.attrs["boundary_resolution"] = boundaries.attrs.get("boundary_resolution", "raw")
    merged.attrs["data_source_mode"] = data_source_mode
    merged.attrs["data_last_updated"] = _data_last_updated(gold_risk)
    merged.attrs["model_version"] = (
        gold_risk["model_version"].dropna().unique().tolist() if not gold_risk.empty and "model_version" in gold_risk.columns else []
    )

    return merged, shap_df


def latest_year(df: pd.DataFrame) -> int | None:
    if "year" not in df.columns or df["year"].dropna().empty:
        return None
    return int(pd.to_numeric(df["year"], errors="coerce").dropna().max())


def _filter_state(df: pd.DataFrame, state_filter: str | None) -> pd.DataFrame:
    if not state_filter or state_filter == "All Nigeria":
        return df
    return df[df["state_name"] == state_filter]


def _filter_year(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is None or "year" not in df.columns:
        return df
    return df[df["year"] == year]


def _collapse_years(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across years by averaging numeric columns, keeping first geometry + labels."""
    if df.empty:
        return df
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    keep_first = [col for col in df.columns if col not in numeric_cols]
    agg = {col: "mean" for col in numeric_cols}
    for col in keep_first:
        agg[col] = "first"
    collapsed = df.groupby("lga_uid", as_index=False).agg(agg)
    return collapsed


def filter_geo(geo_df: pd.DataFrame, state_filter: str | None = None, year: str | int | None = None) -> pd.DataFrame:
    """Convenience wrapper for state/year filtering, supports 'Both' to aggregate across years."""
    filtered = _filter_state(geo_df, state_filter)
    if str(year).lower() == "both":
        return _collapse_years(filtered)
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return _filter_year(filtered, year_int)


def normalize_for_choropleth(gdf: gpd.GeoDataFrame, column: str) -> list[float]:
    """Return normalized 0-1 values for coloring, defaulting to 0.5 when missing."""
    series = pd.to_numeric(gdf[column], errors="coerce") if column in gdf.columns else pd.Series([], dtype=float)
    if series.empty or series.notna().sum() == 0:
        return [0.5] * len(gdf)
    min_v, max_v = series.min(), series.max()
    if min_v == max_v:
        return [0.5] * len(gdf)
    return [
        float((val - min_v) / (max_v - min_v)) if pd.notna(val) else 0.5
        for val in series.reindex(gdf.index, fill_value=np.nan)
    ]


def get_states(geo_df: pd.DataFrame) -> list[str]:
    return sorted(geo_df["state_name"].dropna().astype(str).unique().tolist())


def get_lgas_geojson(
    geo_df: gpd.GeoDataFrame,
    state_filter: str | None = None,
    year: str | int | None = None,
    columns: Iterable[str] = ("lga_uid", "lga_name", "state_name", "risk_score"),
) -> str:
    filtered = filter_geo(geo_df, state_filter, year)
    columns = [c for c in columns if c in filtered.columns] + ["geometry"]
    trimmed = filtered[columns]
    return trimmed.to_crs(epsg=4326).to_json()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_ranked_hotspots(
    geo_df: pd.DataFrame,
    focus: str,
    state_filter: str | None = None,
    year: str | int | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    column = FOCUS_COLUMN.get(focus, "risk_score")
    ascending = column in ASCENDING_FOCUS
    filtered = filter_geo(geo_df, state_filter, year)
    if column not in filtered.columns:
        filtered[column] = np.nan
    ranked = filtered.sort_values(column, ascending=ascending, na_position="last").head(limit)
    ranked = ranked.reset_index(drop=True)
    hotspots: list[dict[str, Any]] = []
    for idx, row in ranked.iterrows():
        hotspots.append(
            {
                "rank": int(idx + 1),
                "id": str(row.get("lga_uid")),
                "name": row.get("lga_name"),
                "state": row.get("state_name"),
                "risk": _safe_float(row.get("risk_score")),
                "risk_total": _safe_float(row.get("risk_score_total")),
                "fac": _safe_float(row.get("facilities_per_10k")),
                "dist": _safe_float(row.get("avg_distance_km")),
                "u5mr": _safe_float(row.get("u5mr_mean")),
                "pop": _safe_float(row.get("population")),
                "cov": _safe_float(row.get("coverage_5km")),
                "towers": _safe_float(row.get("towers_per_10k")),
                "year": row.get("year"),
                "confidence_pct": _safe_float(row.get("confidence_pct")),
                "confidence_reason_codes": row.get("confidence_reason_codes"),
                "primary_barriers": row.get("primary_barriers"),
                "recommendation": row.get("recommendation"),
            }
        )
    return hotspots


def get_lga_detail(
    geo_df: pd.DataFrame,
    shap_df: pd.DataFrame | None,
    lga_uid: str,
    year: str | int | None = None,
) -> dict[str, Any] | None:
    filtered = filter_geo(geo_df, None, year)
    match = filtered[filtered["lga_uid"].astype(str) == str(lga_uid)]
    if match.empty:
        return None
    row = match.iloc[0]
    shap_values = get_shap_values(shap_df, row.get("lga_name"), row.get("year"))
    return {
        "id": str(row.get("lga_uid")),
        "name": row.get("lga_name"),
        "state": row.get("state_name"),
        "risk": _safe_float(row.get("risk_score")),
        "risk_total": _safe_float(row.get("risk_score_total")),
        "fac": _safe_float(row.get("facilities_per_10k")),
        "dist": _safe_float(row.get("avg_distance_km")),
        "u5mr": _safe_float(row.get("u5mr_mean")),
        "pop": _safe_float(row.get("population")),
        "cov": _safe_float(row.get("coverage_5km")),
        "towers": _safe_float(row.get("towers_per_10k")),
        "density": _safe_float(row.get("population_density")),
        "year": row.get("year"),
        "confidence_pct": _safe_float(row.get("confidence_pct")),
        "confidence_reason_codes": row.get("confidence_reason_codes"),
        "primary_barriers": row.get("primary_barriers"),
        "recommendation": row.get("recommendation"),
        "shap": shap_values,
    }


def get_shap_values(
    shap_df: pd.DataFrame | None,
    lga_name: str | None,
    year: int | None,
    top_n: int = 8,
) -> dict[str, float] | None:
    if shap_df is None or lga_name is None:
        return None
    subset = shap_df[shap_df["lga_name"] == lga_name]
    if "year" in shap_df.columns and year is not None:
        subset = subset[subset["year"] == year]
    if subset.empty:
        return None
    row = subset.iloc[0]
    drop_cols = [c for c in ["lga_name", "year"] if c in row.index]
    values = row.drop(labels=drop_cols)
    values = values[values.notna()]
    if values.empty:
        return None
    values = values.reindex(values.abs().sort_values(ascending=False).index)
    return {k: float(v) for k, v in values.head(top_n).items()}
