"""Build LGA-level features from DHS clusters and facility data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import pyreadstat
except ImportError:  # pragma: no cover - pandas fallback remains supported
    pyreadstat = None

from scripts.export_dhs_clusters import compute_cluster_u5mr_from_br, prepare_birth_records
from src.config import (
    AVG_DISTANCE_KM_MAX,
    COVERAGE_KM_DEFAULT,
    MIN_LGA_COUNT,
    POPULATION_MERGE_COVERAGE_MIN,
)
from src.data.spatial_ops import (
    CRS,
    aggregate_facility_metrics_by_lga,
    aggregate_tower_metrics_by_lga,
    assign_points_to_lga,
    coverage_within_km,
    infer_lga_names_from_facilities,
    load_facilities,
    load_lga_boundaries,
    load_opencellid,
    make_points_from_latlon,
    normalize_admin_name,
)

DHS_YEAR_CONFIG = {
    2013: {
        "kr_dir": "NGKR6ADT",
        "hr_dir": "NGHR6ADT",
        "gps_dir": "NGGE6AFL",
        "cov_dir": None,
    },
    2018: {
        "kr_dir": "NGKR7BDT",
        "hr_dir": "NGHR7BDT",
        "gps_dir": "NGGE7BFL",
        "cov_dir": None,
    },
    2024: {
        "kr_dir": "NGKR8BDT",
        "hr_dir": "NGHR8BDT",
        "gps_dir": "NGGE8AFL",
        "cov_dir": "NGGC8AFL",
    },
}
PRESERVE_EXISTING_YEARS = {2013, 2018}
SUPPORTED_YEARS = tuple(DHS_YEAR_CONFIG.keys())

KR_REQUIRED_COLUMNS = ["v001", "v002", "v005", "v008", "b3", "b5", "b7", "v024", "v025"]
HR_SELECT_COLUMNS = ["v001", "v002", "v005", "v024", "v190", "hv001", "hv002", "hv005", "hv024", "hv270", "hv201", "hv205"]
EXTRA_OUTPUT_COLUMNS = [
    "wealth_index_mean",
    "water_source_mean",
    "toilet_type_mean",
    "urban_rural_dhs",
    "travel_time_city_min",
    "altitude_m",
]


def _configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/build_features.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _normalize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cluster column names and types."""

    lat_candidates = ("latitude", "lat", "LAT", "LATITUDE", "cluster_lat", "y")
    lon_candidates = ("longitude", "lon", "LON", "LONGITUDE", "cluster_lon", "x")
    urban_candidates = ("urban_rural", "urban", "URBAN", "is_urban")

    lower_map = {col.lower(): col for col in df.columns}
    lat_src = next((lower_map.get(c.lower()) for c in lat_candidates if c.lower() in lower_map), None)
    lon_src = next((lower_map.get(c.lower()) for c in lon_candidates if c.lower() in lower_map), None)
    urban_src = next((lower_map.get(c.lower()) for c in urban_candidates if c.lower() in lower_map), None)

    if not lat_src or not lon_src:
        first_row = df.head(1).to_dict(orient="records")
        raise ValueError(
            f"Could not locate latitude/longitude columns in clusters CSV. "
            f"Columns: {list(df.columns)}; first row: {first_row}"
        )

    df = df.rename(columns={lat_src: "latitude", lon_src: "longitude"})
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")

    if urban_src:
        df = df.rename(columns={urban_src: "urban"}) if urban_src != "urban" else df
        urban_series = df["urban"]
        if urban_series.dtype == bool:
            df["urban"] = urban_series.astype(int)
        elif pd.api.types.is_numeric_dtype(urban_series):
            df["urban"] = pd.to_numeric(urban_series, errors="coerce")
        else:
            mapped = (
                urban_series.astype(str)
                .str.strip()
                .str.upper()
                .map({"URBAN": 1, "U": 1, "1": 1, "RURAL": 0, "R": 0, "2": 0})
            )
            df["urban"] = mapped
    logging.info(
        "Mapped %s->latitude, %s->longitude%s",
        lat_src,
        lon_src,
        f", {urban_src}->urban" if urban_src else ", no urban column mapped",
    )
    return df


def _norm_key(series: pd.Series) -> pd.Series:
    """Normalize string keys for deterministic joining."""

    return normalize_admin_name(series)


def _load_population(
    population_path: Path | None,
    pop_lga_col: str | None = None,
    pop_state_col: str | None = None,
) -> pd.DataFrame | None:
    """Load population CSV if present and normalize column names."""

    if not population_path or not population_path.exists():
        return None

    population_df = pd.read_csv(population_path)
    rename_map = {}
    if pop_lga_col and pop_lga_col in population_df.columns:
        rename_map[pop_lga_col] = "lga_name"
    if pop_state_col and pop_state_col in population_df.columns:
        rename_map[pop_state_col] = "state_name"
    if rename_map:
        population_df = population_df.rename(columns=rename_map)

    expected = {"lga_name", "state_name", "population"}
    missing_cols = expected - set(population_df.columns)
    if missing_cols:
        raise ValueError(f"Population file missing required columns: {missing_cols}")

    population_df["lga_name"] = population_df["lga_name"].astype(str).fillna("")
    population_df["state_name"] = population_df["state_name"].astype(str).fillna("")
    population_df["state_lga_norm"] = _norm_key(population_df["state_name"]) + "__" + _norm_key(population_df["lga_name"])
    return population_df


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: rigorous validation of the feature table."""

    key_cols = ["lga_uid", "year"] if "year" in df.columns else ["lga_uid"]
    if df[key_cols].duplicated().any():
        dup_keys = df.loc[df[key_cols].duplicated(), key_cols].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate feature keys found. Sample duplicates: {dup_keys}")

    dup_lga_names = df["lga_name"].value_counts()
    dup_lga_names = dup_lga_names[dup_lga_names > 1]
    if not dup_lga_names.empty:
        logging.info(
            "Found %d duplicated lga_name values; top: %s",
            len(dup_lga_names),
            dup_lga_names.head(10).to_dict(),
        )

    if len(df) < MIN_LGA_COUNT:
        logging.warning("Row count (%d) below expected Nigeria LGA count.", len(df))

    if not df["avg_distance_km"].between(0, AVG_DISTANCE_KM_MAX).all():
        raise ValueError(f"avg_distance_km values out of expected range 0-{AVG_DISTANCE_KM_MAX} km.")

    if (df["facilities_per_10k"] < 0).any():
        raise ValueError("facilities_per_10k contains negative values.")

    if (df["coverage_5km"] < 0).any() or (df["coverage_5km"] > 100).any():
        violators = df[(df["coverage_5km"] < 0) | (df["coverage_5km"] > 100)][["lga_uid", "coverage_5km"]].head(10)
        raise ValueError(f"coverage_5km must be within [0, 100]. Sample violators:\n{violators.to_string()}")

    if "population_density" in df and (df["population_density"] < 0).any():
        bad = df[df["population_density"] < 0][["lga_uid", "population_density"]].head(10)
        raise ValueError(f"population_density must be non-negative. Sample:\n{bad.to_string()}")

    core_fields = ["u5mr_mean", "facilities_per_10k", "avg_distance_km", "coverage_5km"]
    null_rates = {col: df[col].isna().mean() for col in core_fields if col in df}
    threshold = 0.5
    over = {col: rate for col, rate in null_rates.items() if rate > threshold}
    if over:
        raise ValueError(f"Core field null rates too high (>{threshold:.0%}): {over}")

    labeled = df["u5mr_mean"].notna().sum()
    logging.info(
        "Label coverage: %d of %d rows (%.1f%%) have u5mr_mean",
        labeled,
        len(df),
        labeled / len(df) * 100 if len(df) else 0.0,
    )
    return df


def _build_report(
    df: pd.DataFrame,
    output_path: Path,
    tower_rows_loaded: int = 0,
    tower_lga_coverage: float = 0.0,
) -> None:
    report = {
        "rows": len(df),
        "years": sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique().tolist())
        if "year" in df.columns
        else [],
        "avg_distance_km_mean": float(df["avg_distance_km"].mean()),
        "u5mr_mean_mean": float(df["u5mr_mean"].mean()),
        "facilities_count_total": int(df["facilities_count"].sum()),
        "population_rows_with_values": int(df["population"].notna().sum()),
        "tower_rows_loaded": int(tower_rows_loaded),
        "tower_lga_coverage_pct": float(tower_lga_coverage),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))


def _prepare_lga_context(
    boundaries_path: Path,
    facilities_path: Path,
    lga_col: str | None,
    state_col: str | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    lgas = load_lga_boundaries(str(boundaries_path), lga_col=lga_col, state_col=state_col)
    facilities = load_facilities(str(facilities_path))

    need_inference = (
        "lga_name" not in lgas.columns
        or lgas["lga_name"].isna().all()
        or lgas["lga_name"].astype(str).str.strip().eq("").all()
    )
    if need_inference or lgas["lga_name"].isna().any() or lgas["lga_name"].astype(str).str.strip().eq("").any():
        logging.info("Inferring LGA names via facilities overlay.")
        lgas = infer_lga_names_from_facilities(lgas, facilities)
        non_placeholder_fraction = (~lgas["lga_name"].astype(str).str.startswith("LGA_")).mean()
        if non_placeholder_fraction < 0.5:
            raise ValueError(
                "Fewer than 50% of polygons received non-placeholder LGA names after inference; "
                "the boundary file may not match the facilities geography."
            )

    lgas = lgas.copy()
    if "lga_id" not in lgas:
        lgas["lga_id"] = np.arange(len(lgas))
    lgas["lga_uid"] = lgas["lga_id"]
    lgas["state_name"] = lgas.get("state_name", pd.Series(index=lgas.index, dtype=object)).fillna("")
    lgas["lga_name"] = lgas["lga_name"].astype(str)
    state_clean = lgas["state_name"].astype(str).str.strip()
    lga_clean = lgas["lga_name"].astype(str).str.strip()
    lgas["state_lga"] = state_clean + "__" + lga_clean
    lgas["state_lga_norm"] = _norm_key(state_clean) + "__" + _norm_key(lga_clean)
    return lgas, facilities


def ingest_and_normalize(
    dhs_path: Path,
    boundaries_path: Path,
    facilities_path: Path,
    lga_col: str | None,
    state_col: str | None,
):
    """Stage 1: load raw files and normalize columns / identifiers."""

    logging.info("Loading DHS clusters from %s", dhs_path)
    clusters = _normalize_clusters(pd.read_csv(dhs_path))
    if "cluster_id" not in clusters.columns:
        clusters["cluster_id"] = np.arange(1, len(clusters) + 1)
    clusters_gdf = make_points_from_latlon(clusters, lat_col="latitude", lon_col="longitude")
    lgas, facilities = _prepare_lga_context(boundaries_path, facilities_path, lga_col, state_col)
    return clusters_gdf, lgas, facilities


def join_and_aggregate(
    clusters_gdf: gpd.GeoDataFrame,
    lgas_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    coverage_km: float,
) -> pd.DataFrame:
    """Stage 2: spatial joins and core aggregations (no side effects)."""

    joined = assign_points_to_lga(clusters_gdf, lgas_gdf)
    if "u5mr" not in joined.columns:
        joined["u5mr"] = joined["u5_deaths"] / joined["live_births"] * 1000.0
    else:
        fallback = joined["u5_deaths"] / joined["live_births"] * 1000.0
        joined["u5mr"] = pd.to_numeric(joined["u5mr"], errors="coerce").fillna(fallback)
    joined["lga_uid"] = joined["lga_id"]

    outcomes = (
        joined.groupby("lga_uid")
        .agg(
            u5mr_mean=("u5mr", "mean"),
            u5mr_median=("u5mr", "median"),
            live_births_sum=("live_births", "sum"),
            u5_deaths_sum=("u5_deaths", "sum"),
            urban_prop=("urban", "mean"),
            lga_lat=("lat", "mean"),
            lga_lon=("lon", "mean"),
        )
        .reset_index()
    )

    base = lgas_gdf[["lga_uid", "state_name", "lga_name", "state_lga", "state_lga_norm"]].drop_duplicates("lga_uid")
    features = base.merge(outcomes, on="lga_uid", how="left")

    facilities_metrics = aggregate_facility_metrics_by_lga(facilities_gdf, lgas_gdf, population_df=None)
    facilities_metrics = facilities_metrics.rename(
        columns={"avg_distance_km_proxy": "avg_distance_km", "lga_id": "lga_uid"}
    )
    facilities_metrics = facilities_metrics[
        ["lga_uid", "avg_distance_km", "facilities_count", "facilities_per_10k"]
    ]

    coverage_df = coverage_within_km(lgas_gdf, facilities_gdf, km=coverage_km)
    if "population_covered_pct" in coverage_df.columns:
        coverage_df = coverage_df.rename(columns={"population_covered_pct": "coverage_5km"})
    else:
        coverage_df = coverage_df.rename(columns={"area_covered_pct": "coverage_5km"})
    coverage_df = coverage_df.rename(columns={"lga_id": "lga_uid"})

    _cov_vals = coverage_df["coverage_5km"].dropna()
    if (_cov_vals < 0).any() or (_cov_vals > 100).any():
        bad = coverage_df[
            (coverage_df["coverage_5km"] < 0) | (coverage_df["coverage_5km"] > 100)
        ][["lga_uid", "lga_name", "coverage_5km"]].head(10)
        raise ValueError(
            f"coverage_5km out of valid [0, 100] range after clamping. "
            f"Sample violators:\n{bad.to_string()}\n"
            "Check facility geometry validity and LGA boundary overlaps."
        )

    features = (
        features.merge(facilities_metrics, on="lga_uid", how="left")
        .merge(coverage_df[["lga_uid", "coverage_5km"]], on="lga_uid", how="left")
    )
    return features


def enrich_and_impute(
    features_df: pd.DataFrame,
    lgas_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame | None,
    towers_gdf: gpd.GeoDataFrame | None,
):
    """Stage 3: merge population/tower data, derive densities, and impute gaps."""

    features = features_df.copy()
    lgas = lgas_gdf.copy()

    lga_centroids = lgas[["lga_uid", "geometry"]].copy().to_crs(CRS.metric)
    lga_centroids["centroid_geom"] = lga_centroids.geometry.centroid
    lga_centroids = lga_centroids.set_geometry("centroid_geom").to_crs(CRS.wgs84)
    lga_centroids["lga_lat_centroid"] = lga_centroids.geometry.y
    lga_centroids["lga_lon_centroid"] = lga_centroids.geometry.x
    features = features.merge(
        lga_centroids[["lga_uid", "lga_lat_centroid", "lga_lon_centroid"]],
        on="lga_uid",
        how="left",
    )
    for col, cent_col in [("lga_lat", "lga_lat_centroid"), ("lga_lon", "lga_lon_centroid")]:
        if col in features:
            features[col] = features[col].fillna(features[cent_col])
    features = features.drop(columns=["lga_lat_centroid", "lga_lon_centroid"])

    def _impute_by_state(df: pd.DataFrame, col: str, func: str = "median"):
        state_stat = df.groupby("state_name")[col].transform(func)
        overall = getattr(df[col], func)()
        return df[col].fillna(state_stat).fillna(overall)

    for col, func in [("u5mr_mean", "mean"), ("u5mr_median", "median"), ("urban_prop", "mean")]:
        if col in features:
            features[col] = _impute_by_state(features, col, func)

    for col in ["live_births_sum", "u5_deaths_sum"]:
        if col in features:
            features[col] = features[col].fillna(0)

    tower_rows_loaded = 0
    tower_lga_coverage = 0.0
    if towers_gdf is not None:
        tower_rows_loaded = len(towers_gdf)
        tower_metrics = aggregate_tower_metrics_by_lga(towers_gdf, lgas)
        tower_lga_coverage = (tower_metrics["towers_count"] > 0).mean() * 100 if len(tower_metrics) else 0.0
        features = features.merge(tower_metrics, on="lga_uid", how="left")
        for col in ["towers_count", "tower_density_per_km2", "avg_dist_to_tower_km"]:
            if col in features:
                features[col] = features[col].fillna(0 if "count" in col or "density" in col else np.nan)
    else:
        features["towers_count"] = 0
        features["tower_density_per_km2"] = 0.0
        features["avg_dist_to_tower_km"] = np.nan

    # --- Travel time coverage (ORS isochrones, optional) ---
    travel_time_path = Path("data/raw/travel_time_lga.csv")
    if travel_time_path.exists():
        tt = pd.read_csv(travel_time_path)
        # Normalise join key to match features lga_id slug format
        needs_slug_key = (
            "lga_id" not in tt.columns
            or not tt["lga_id"].astype(str).str.contains("__", regex=False).all()
        )
        if needs_slug_key and {"lga_name", "state_name"}.issubset(tt.columns):
            lga_slug = tt["lga_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            state_slug = tt["state_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            tt["lga_id"] = state_slug + "__" + lga_slug
        tt_cols = [c for c in [
            "lga_id",
            "pop_pct_30min",
            "pop_pct_60min",
            "pop_pct_120min",
            "pop_pct_within_60min",
            "pop_covered_60min",
        ] if c in tt.columns]
        if "lga_id" in tt_cols:
            # Use a normalized state/LGA slug as the join bridge; travel-time is year-agnostic.
            feature_lga_id = (
                features["state_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
                + "__"
                + features["lga_name"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
            )
            features = features.assign(_travel_lga_id=feature_lga_id).merge(
                tt[tt_cols],
                left_on="_travel_lga_id",
                right_on="lga_id",
                how="left",
            )
            features = features.drop(columns=[c for c in ["_travel_lga_id", "lga_id"] if c in features.columns])
            logging.info(
                "Travel time merge: %d LGAs with pop_pct_60min",
                features["pop_pct_60min"].notna().sum() if "pop_pct_60min" in features.columns else 0,
            )
    else:
        for col in ["pop_pct_30min", "pop_pct_60min", "pop_pct_120min", "pop_pct_within_60min", "pop_covered_60min"]:
            features[col] = np.nan
        logging.info("Travel time file not found at %s — columns set to NaN.", travel_time_path)

    lga_area = lgas[["lga_uid", "geometry"]].copy().to_crs(CRS.metric)
    lga_area["area_sq_km"] = lga_area.geometry.area / 1_000_000.0
    features = features.merge(lga_area[["lga_uid", "area_sq_km"]], on="lga_uid", how="left")

    if population_df is not None and "population" in population_df.columns:
        pop_df = population_df.copy()
        if {"state_name", "lga_name"}.issubset(pop_df.columns):
            pop_df["state_lga_norm"] = _norm_key(pop_df["state_name"]) + "__" + _norm_key(pop_df["lga_name"])
            merge_cols = ["state_lga_norm", "population"]
            if "area_sq_km" in pop_df.columns:
                merge_cols.append("area_sq_km")
            features = features.merge(pop_df[merge_cols], on="state_lga_norm", how="left")
        else:
            merge_cols = ["lga_name", "population"]
            if "area_sq_km" in pop_df.columns:
                merge_cols.append("area_sq_km")
            features = features.merge(pop_df[merge_cols], on="lga_name", how="left")
        if "area_sq_km" in features.columns:
            features["population_density"] = features["population"] / features["area_sq_km"]
        else:
            features["population_density"] = np.nan

        matched = features["population"].notna().sum()
        unmatched_keys = features.loc[features["population"].isna(), "state_lga_norm"].head(10).tolist()
        coverage = matched / len(features) if len(features) else 0.0
        logging.info(
            "Population merge: matched %d / %d rows (%.1f%%). Sample unmatched keys: %s",
            matched,
            len(features),
            coverage * 100,
            unmatched_keys,
        )
        min_pop = features["population"].min(skipna=True)
        max_pop = features["population"].max(skipna=True)
        zero_pop = ((features["population"] == 0) & features["population"].notna()).sum()
        logging.info(
            "Population stats: min=%.0f max=%.0f zero_pop_rows=%d",
            min_pop if pd.notna(min_pop) else float("nan"),
            max_pop if pd.notna(max_pop) else float("nan"),
            zero_pop,
        )
        if coverage < POPULATION_MERGE_COVERAGE_MIN:
            raise ValueError(
                f"Population merge coverage too low: {coverage:.1%}. "
                f"Sample unmatched keys: {unmatched_keys}"
            )
    else:
        features["population"] = np.nan
        features["population_density"] = np.nan

    if features["population"].notna().any():
        features["facilities_per_10k"] = np.where(
            features["population"] > 0,
            features["facilities_count"] / (features["population"] / 10000.0),
            np.nan,
        )
        if "towers_count" in features.columns:
            features["towers_per_10k_pop"] = np.where(
                features["population"] > 0,
                features["towers_count"] / (features["population"] / 10000.0),
                np.nan,
            )

    def _pct_rank(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.rank(pct=True)

    facilities_pct = _pct_rank(features.get("facilities_per_10k", np.nan))
    distance_pct = _pct_rank(features.get("avg_distance_km", np.nan))
    if distance_pct.notna().sum() == 0:
        features["access_score"] = 1 - facilities_pct
    else:
        features["access_score"] = (1 - facilities_pct + distance_pct) / 2

    return features, {"tower_rows_loaded": tower_rows_loaded, "tower_lga_coverage": tower_lga_coverage}


def persist_features(
    features_df: pd.DataFrame,
    lgas_gdf: gpd.GeoDataFrame,
    output_path: Path,
    report_path: Path,
    tower_rows_loaded: int = 0,
    tower_lga_coverage: float = 0.0,
):
    """Stage 5: reorder schema and write outputs."""

    ordered_cols = [
        "lga_uid",
        "lga_name",
        "state_name",
        "state_lga",
        "area_sq_km",
        "lga_lat",
        "lga_lon",
        "u5mr_mean",
        "u5mr_median",
        "live_births_sum",
        "u5_deaths_sum",
        "facilities_count",
        "facilities_per_10k",
        "avg_distance_km",
        "urban_prop",
        "population",
        "population_density",
        "access_score",
        "towers_count",
        "tower_density_per_km2",
        "avg_dist_to_tower_km",
        "towers_per_10k_pop",
        "coverage_5km",
        "pop_pct_30min",
        "pop_pct_60min",
        "pop_pct_120min",
        "pop_pct_within_60min",
        "pop_covered_60min",
        "year",
        *EXTRA_OUTPUT_COLUMNS,
    ]
    features = features_df.copy()
    for col in ordered_cols:
        if col not in features.columns:
            features[col] = np.nan
    trailing = [col for col in features.columns if col not in ordered_cols]
    features = features[ordered_cols + trailing]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    _build_report(
        features,
        report_path,
        tower_rows_loaded=tower_rows_loaded,
        tower_lga_coverage=tower_lga_coverage,
    )
    logging.info(
        "OK: built features with %d rows | years=%s | towers_mean=%.2f | pct_rows_with_tower=%.1f%%",
        len(features),
        sorted(pd.to_numeric(features["year"], errors="coerce").dropna().astype(int).unique().tolist())
        if "year" in features.columns
        else [],
        features["towers_count"].mean() if "towers_count" in features else 0,
        (features["towers_count"] > 0).mean() * 100 if "towers_count" in features else 0,
    )
    print(
        f"OK: {len(features)} rows | avg_distance_km mean={features['avg_distance_km'].mean():.2f} | "
        f"u5mr_mean mean={features['u5mr_mean'].mean():.2f} | "
        f"years={sorted(pd.to_numeric(features['year'], errors='coerce').dropna().astype(int).unique().tolist())}"
    )
    return {"csv": output_path, "report": report_path}


def _build_single_source_features(
    clusters_path: Path,
    lga_path: Path,
    facilities_path: Path,
    population_path: Path | None,
    output_path: Path,
    report_path: Path,
    coverage_km: float,
    lga_col: str | None = None,
    state_col: str | None = None,
    pop_lga_col: str | None = None,
    pop_state_col: str | None = None,
    opencellid_path: Path | None = Path("data/raw/opencellid.csv.gz"),
) -> pd.DataFrame:
    clusters_gdf, lgas_gdf, facilities_gdf = ingest_and_normalize(
        clusters_path, lga_path, facilities_path, lga_col, state_col
    )

    population_df = _load_population(population_path, pop_lga_col=pop_lga_col, pop_state_col=pop_state_col)

    towers_gdf = None
    if opencellid_path and Path(opencellid_path).exists():
        towers_gdf = load_opencellid(str(opencellid_path))

    features_stage2 = join_and_aggregate(clusters_gdf, lgas_gdf, facilities_gdf, coverage_km)
    features_stage3, meta = enrich_and_impute(features_stage2, lgas_gdf, population_df, towers_gdf)
    if "year" not in features_stage3.columns:
        features_stage3["year"] = 2018
    validated = validate_features(features_stage3)
    persist_features(
        validated,
        lgas_gdf,
        output_path,
        report_path,
        tower_rows_loaded=meta.get("tower_rows_loaded", 0),
        tower_lga_coverage=meta.get("tower_lga_coverage", 0.0),
    )
    return validated


def _discover_dhs_file(directory: Path, suffixes: tuple[str, ...]) -> Path | None:
    if not directory.exists():
        return None
    suffixes = tuple(s.lower() for s in suffixes)
    candidates = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes],
        key=lambda item: (suffixes.index(item.suffix.lower()), item.name.lower()),
    )
    return candidates[0] if candidates else None


def _discover_dhs_inputs(year: int) -> dict[str, Path | None]:
    root = Path("data/raw/dhs") / str(year)
    cfg = DHS_YEAR_CONFIG[year]
    return {
        "kr": _discover_dhs_file(root / cfg["kr_dir"], (".dta", ".csv")),
        "hr": _discover_dhs_file(root / cfg["hr_dir"], (".dta", ".csv")),
        "gps": _discover_dhs_file(root / cfg["gps_dir"], (".shp",)),
        "covariates": _discover_dhs_file(root / cfg["cov_dir"], (".csv",)) if cfg["cov_dir"] else None,
    }


def _available_table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    reader = pd.read_stata(path, convert_categoricals=False, iterator=True)
    return reader.read(1).columns.tolist()


def _read_dhs_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    available = _available_table_columns(path)
    selected = [col for col in (usecols or available) if col in available]
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=selected or None)
    if pyreadstat is not None:
        try:
            df, _ = pyreadstat.read_dta(str(path), usecols=selected or None)
            return df
        except Exception as exc:  # pragma: no cover - exercised only when pyreadstat is installed
            logging.warning("pyreadstat failed for %s (%s); falling back to pandas.read_stata.", path, exc)
    return pd.read_stata(path, columns=selected or None, convert_categoricals=False)


def _select_with_aliases(
    df: pd.DataFrame,
    alias_map: dict[str, tuple[str, ...]],
    *,
    dataset_label: str,
    year: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for target, candidates in alias_map.items():
        source = next((candidate for candidate in candidates if candidate in df.columns), None)
        if source is None:
            logging.warning("%s %s missing %s; filling NaN.", dataset_label, year, target)
            out[target] = np.nan
        else:
            out[target] = df[source]
    return out


def _urban_to_numeric(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.map({1: 1.0, 2: 0.0}).fillna(numeric)
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"U": 1.0, "URBAN": 1.0, "1": 1.0, "R": 0.0, "RURAL": 0.0, "2": 0.0})
    )


def _load_cluster_points(gps_path: Path | None, mock_clusters_path: Path | None) -> gpd.GeoDataFrame:
    if gps_path and gps_path.exists():
        gdf = gpd.read_file(gps_path)
        cols = {col.upper(): col for col in gdf.columns}
        cluster_col = cols.get("DHSCLUST")
        lat_col = cols.get("LATNUM") or cols.get("LATITUDE") or cols.get("LAT")
        lon_col = cols.get("LONGNUM") or cols.get("LONNUM") or cols.get("LONGITUDE") or cols.get("LON")
        urban_col = cols.get("URBAN_RURA") or cols.get("URBAN_RUR") or cols.get("URBAN_R")
        if not cluster_col or not lat_col or not lon_col:
            raise ValueError(f"GPS file missing required cluster/lat/lon columns: {gps_path}")
        points_df = pd.DataFrame(
            {
                "cluster_id": pd.to_numeric(gdf[cluster_col], errors="coerce").astype("Int64"),
                "latitude": pd.to_numeric(gdf[lat_col], errors="coerce"),
                "longitude": pd.to_numeric(gdf[lon_col], errors="coerce"),
                "urban_rural_source": gdf[urban_col] if urban_col else pd.NA,
            }
        )
    elif mock_clusters_path and mock_clusters_path.exists():
        points_df = _normalize_clusters(pd.read_csv(mock_clusters_path))
        if "cluster_id" not in points_df.columns:
            points_df["cluster_id"] = np.arange(1, len(points_df) + 1)
        points_df = points_df.rename(columns={"urban": "urban_rural_source"})
        points_df["urban_rural_source"] = points_df["urban_rural_source"].map({1: "U", 0: "R"}).fillna(points_df["urban_rural_source"])
        points_df = points_df[["cluster_id", "latitude", "longitude", "urban_rural_source"]]
        logging.warning("GPS shapefile missing; using %s as mock cluster coordinate fallback.", mock_clusters_path)
    else:
        raise FileNotFoundError("No DHS GPS shapefile or mock cluster CSV available.")

    points_df = points_df.dropna(subset=["cluster_id", "latitude", "longitude"]).copy()
    points_df["cluster_id"] = pd.to_numeric(points_df["cluster_id"], errors="coerce").astype("Int64")
    points_df["urban_rural_flag"] = _urban_to_numeric(points_df["urban_rural_source"])
    return make_points_from_latlon(points_df, lat_col="latitude", lon_col="longitude")


def _cluster_to_lga_mapping(cluster_points: gpd.GeoDataFrame, lgas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    join_cols = ["lga_uid", "lga_name", "geometry"]
    if "state_name" in lgas_gdf.columns:
        join_cols.insert(2, "state_name")
    joined = gpd.sjoin(cluster_points, lgas_gdf[join_cols], how="left", predicate="within")
    join_rate = joined["lga_uid"].notna().mean() if len(joined) else 0.0
    logging.info("DHS cluster->LGA spatial join success rate: %.1f%%", join_rate * 100)
    if join_rate < 0.9:
        logging.warning("Cluster spatial join success rate below 90%%. Unmatched clusters: %d", joined["lga_uid"].isna().sum())
    mapping = joined.drop(columns=[col for col in ["index_right"] if col in joined.columns]).copy()
    mapping["lat"] = mapping.geometry.y
    mapping["lon"] = mapping.geometry.x
    return mapping.drop_duplicates(subset=["cluster_id"])


def _empty_cluster_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_id",
            "live_births",
            "u5_deaths",
            "u5mr",
            "region",
            "urban_code",
            "_ur_fallback",
            "quality_flag",
        ]
    )


def _compute_kr_cluster_metrics(kr_df: pd.DataFrame, year: int) -> pd.DataFrame:
    alias_map = {col: (col,) for col in KR_REQUIRED_COLUMNS}
    normalized = _select_with_aliases(kr_df, alias_map, dataset_label="KR", year=year)
    for col in KR_REQUIRED_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    prepped_df, _ = prepare_birth_records(normalized, {})
    if prepped_df.empty:
        logging.warning("KR %s contains no births in the 5-year window after filtering.", year)
        return _empty_cluster_metrics()
    cluster_metrics = compute_cluster_u5mr_from_br(prepped_df)
    if cluster_metrics.empty:
        return _empty_cluster_metrics()
    cluster_metrics["cluster_id"] = pd.to_numeric(cluster_metrics["cluster_id"], errors="coerce").astype("Int64")
    return cluster_metrics


def _aggregate_dhs_clusters_to_lga(
    cluster_df: pd.DataFrame,
    lgas_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    coverage_km: float,
) -> pd.DataFrame:
    base = lgas_gdf[["lga_uid", "state_name", "lga_name", "state_lga", "state_lga_norm"]].drop_duplicates("lga_uid")

    if cluster_df.empty:
        features = base.copy()
        features["u5mr_mean"] = np.nan
        features["u5mr_median"] = np.nan
        features["live_births_sum"] = 0.0
        features["u5_deaths_sum"] = 0.0
        features["urban_prop"] = np.nan
        features["lga_lat"] = np.nan
        features["lga_lon"] = np.nan
    else:
        outcomes = (
            cluster_df.groupby("lga_uid")
            .agg(
                u5mr_mean=("u5mr", "mean"),
                u5mr_median=("u5mr", "median"),
                live_births_sum=("live_births", "sum"),
                u5_deaths_sum=("u5_deaths", "sum"),
                urban_prop=("urban", "mean"),
                lga_lat=("lat", "mean"),
                lga_lon=("lon", "mean"),
            )
            .reset_index()
        )
        features = base.merge(outcomes, on="lga_uid", how="left")

    facilities_metrics = aggregate_facility_metrics_by_lga(facilities_gdf, lgas_gdf, population_df=None)
    facilities_metrics = facilities_metrics.rename(
        columns={"avg_distance_km_proxy": "avg_distance_km", "lga_id": "lga_uid"}
    )[["lga_uid", "avg_distance_km", "facilities_count", "facilities_per_10k"]]

    coverage_df = coverage_within_km(lgas_gdf, facilities_gdf, km=coverage_km)
    if "population_covered_pct" in coverage_df.columns:
        coverage_df = coverage_df.rename(columns={"population_covered_pct": "coverage_5km"})
    else:
        coverage_df = coverage_df.rename(columns={"area_covered_pct": "coverage_5km"})
    coverage_df = coverage_df.rename(columns={"lga_id": "lga_uid"})[["lga_uid", "coverage_5km"]]

    return (
        features.merge(facilities_metrics, on="lga_uid", how="left")
        .merge(coverage_df, on="lga_uid", how="left")
    )


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    mask = numeric_values.notna() & numeric_weights.notna() & (numeric_weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(numeric_values[mask], weights=numeric_weights[mask]))


def _aggregate_household_metrics(hr_df: pd.DataFrame, cluster_map: pd.DataFrame, year: int) -> pd.DataFrame:
    alias_map = {
        "cluster_id": ("v001", "hv001", "cluster_id"),
        "household_id": ("v002", "hv002", "household_id"),
        "sample_weight": ("v005", "hv005", "sample_weight"),
        "state_code": ("v024", "hv024", "state_code"),
        "wealth_index": ("v190", "hv270", "wealth_index"),
        "water_source": ("hv201", "water_source"),
        "toilet_type": ("hv205", "toilet_type"),
    }
    normalized = _select_with_aliases(hr_df, alias_map, dataset_label="HR", year=year)
    normalized["cluster_id"] = pd.to_numeric(normalized["cluster_id"], errors="coerce").astype("Int64")
    normalized["sample_weight"] = pd.to_numeric(normalized["sample_weight"], errors="coerce") / 1_000_000.0

    merged = normalized.merge(cluster_map[["cluster_id", "lga_uid"]].drop_duplicates("cluster_id"), on="cluster_id", how="left")
    merged = merged[merged["lga_uid"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(columns=["lga_uid", "wealth_index_mean", "water_source_mean", "toilet_type_mean"])

    rows = []
    for lga_uid, group in merged.groupby("lga_uid"):
        weights = group["sample_weight"]
        rows.append(
            {
                "lga_uid": lga_uid,
                "wealth_index_mean": _weighted_mean(group["wealth_index"], weights),
                "water_source_mean": _weighted_mean(group["water_source"], weights),
                "toilet_type_mean": _weighted_mean(group["toilet_type"], weights),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_covariate_metrics(covariate_path: Path | None, cluster_map: pd.DataFrame, year: int) -> pd.DataFrame:
    columns = ["lga_uid", "urban_rural_dhs", "travel_time_city_min", "altitude_m"]
    if not covariate_path or not covariate_path.exists():
        return pd.DataFrame(columns=columns)

    cov_df = pd.read_csv(covariate_path)
    if "DHSCLUST" not in cov_df.columns:
        logging.warning("Covariates %s missing DHSCLUST join key; skipping.", covariate_path)
        return pd.DataFrame(columns=columns)

    alias_map = {
        "urban_rural_dhs": ("URBAN_RURA", "URBAN_RURAL"),
        "travel_time_city_min": ("ACCESS_50K", "Travel_Times"),
        "altitude_m": ("SRTM_ALT", "Elevation"),
    }

    extracted = {
        "cluster_id": pd.to_numeric(cov_df["DHSCLUST"], errors="coerce").astype("Int64"),
    }
    for target, candidates in alias_map.items():
        source = next((candidate for candidate in candidates if candidate in cov_df.columns), None)
        if source is None:
            logging.warning("Covariates %s missing %s for year %s; filling NaN.", covariate_path.name, target, year)
            extracted[target] = np.nan
        elif target == "urban_rural_dhs":
            extracted[target] = _urban_to_numeric(cov_df[source])
        else:
            extracted[target] = pd.to_numeric(cov_df[source], errors="coerce")

    extracted_df = pd.DataFrame(extracted)
    merged = extracted_df.merge(cluster_map[["cluster_id", "lga_uid"]].drop_duplicates("cluster_id"), on="cluster_id", how="left")
    merged = merged[merged["lga_uid"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)

    return (
        merged.groupby("lga_uid", as_index=False)[["urban_rural_dhs", "travel_time_city_min", "altitude_m"]]
        .mean()
    )


def _build_dhs_year_features(
    year: int,
    *,
    lgas_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame | None,
    towers_gdf: gpd.GeoDataFrame | None,
    coverage_km: float,
    mock_clusters_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    paths = _discover_dhs_inputs(year)
    if paths["kr"] is None:
        raise FileNotFoundError(f"Missing KR file for DHS {year}.")
    if paths["hr"] is None:
        raise FileNotFoundError(f"Missing HR file for DHS {year}.")

    logging.info("Building DHS features for %s from %s / %s", year, paths["kr"], paths["hr"])
    cluster_points = _load_cluster_points(paths["gps"], mock_clusters_path)
    cluster_map = _cluster_to_lga_mapping(cluster_points, lgas_gdf)

    kr_raw = _read_dhs_table(paths["kr"], usecols=KR_REQUIRED_COLUMNS)
    cluster_metrics = _compute_kr_cluster_metrics(kr_raw, year)
    cluster_features = cluster_map.merge(cluster_metrics, on="cluster_id", how="left")
    cluster_features["urban"] = cluster_features["urban_rural_flag"]
    if "_ur_fallback" in cluster_features.columns:
        cluster_features["urban"] = cluster_features["urban"].fillna(
            cluster_features["_ur_fallback"].map({"U": 1.0, "R": 0.0})
        )
    cluster_features["live_births"] = pd.to_numeric(cluster_features["live_births"], errors="coerce")
    cluster_features["u5_deaths"] = pd.to_numeric(cluster_features["u5_deaths"], errors="coerce")
    cluster_features["u5mr"] = pd.to_numeric(cluster_features["u5mr"], errors="coerce")

    features_stage2 = _aggregate_dhs_clusters_to_lga(cluster_features, lgas_gdf, facilities_gdf, coverage_km)

    hr_raw = _read_dhs_table(paths["hr"], usecols=HR_SELECT_COLUMNS)
    hr_metrics = _aggregate_household_metrics(hr_raw, cluster_map, year)
    cov_metrics = _aggregate_covariate_metrics(paths["covariates"], cluster_map, year)
    features_stage2 = (
        features_stage2.merge(hr_metrics, on="lga_uid", how="left")
        .merge(cov_metrics, on="lga_uid", how="left")
    )

    features_stage3, meta = enrich_and_impute(features_stage2, lgas_gdf, population_df, towers_gdf)
    features_stage3["year"] = year
    return features_stage3, meta


def _load_preserved_rows(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    existing = pd.read_csv(output_path)
    if "year" not in existing.columns:
        return pd.DataFrame()
    return existing


def _build_multiyear_dhs_features(
    *,
    lga_path: Path,
    facilities_path: Path,
    population_path: Path | None,
    output_path: Path,
    report_path: Path,
    coverage_km: float,
    lga_col: str | None,
    state_col: str | None,
    pop_lga_col: str | None,
    pop_state_col: str | None,
    opencellid_path: Path | None,
    mock_clusters_path: Path,
) -> pd.DataFrame:
    lgas_gdf, facilities_gdf = _prepare_lga_context(lga_path, facilities_path, lga_col, state_col)
    population_df = _load_population(population_path, pop_lga_col=pop_lga_col, pop_state_col=pop_state_col)
    towers_gdf = load_opencellid(str(opencellid_path)) if opencellid_path and Path(opencellid_path).exists() else None

    existing = _load_preserved_rows(output_path)
    frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, float]] = []
    preserved_years: set[int] = set()

    if not existing.empty:
        preserved = existing[existing["year"].isin(PRESERVE_EXISTING_YEARS)].copy()
        if not preserved.empty:
            preserved_years = set(pd.to_numeric(preserved["year"], errors="coerce").dropna().astype(int).tolist())
            logging.info(
                "Preserving existing feature rows for years %s without recomputation.",
                sorted(preserved_years),
            )
            frames.append(preserved)

    for year in SUPPORTED_YEARS:
        if year in preserved_years:
            continue
        try:
            year_df, meta = _build_dhs_year_features(
                year,
                lgas_gdf=lgas_gdf,
                facilities_gdf=facilities_gdf,
                population_df=population_df,
                towers_gdf=towers_gdf,
                coverage_km=coverage_km,
                mock_clusters_path=mock_clusters_path,
            )
        except Exception as exc:
            if year == 2024:
                raise
            logging.warning("Could not rebuild DHS year %s from raw inputs; leaving it unchanged if present. %s", year, exc)
            if not existing.empty and (existing["year"] == year).any():
                frames.append(existing[existing["year"] == year].copy())
            continue
        frames.append(year_df)
        meta_rows.append(meta)

    if not frames:
        raise FileNotFoundError("No DHS feature rows could be built or preserved.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["lga_uid", "year"], keep="last")

    tower_rows_loaded = max((int(meta.get("tower_rows_loaded", 0)) for meta in meta_rows), default=0)
    tower_lga_coverage = max((float(meta.get("tower_lga_coverage", 0.0)) for meta in meta_rows), default=0.0)
    validated = validate_features(combined)
    persist_features(
        validated,
        lgas_gdf,
        output_path,
        report_path,
        tower_rows_loaded=tower_rows_loaded,
        tower_lga_coverage=tower_lga_coverage,
    )
    return validated


def build_features(
    clusters_path: Path,
    lga_path: Path,
    facilities_path: Path,
    population_path: Path | None,
    output_path: Path,
    report_path: Path,
    coverage_km: float,
    lga_col: str | None = None,
    state_col: str | None = None,
    pop_lga_col: str | None = None,
    pop_state_col: str | None = None,
    opencellid_path: Path | None = Path("data/raw/opencellid.csv.gz"),
) -> pd.DataFrame:
    """Orchestrate feature builds, preferring multiyear DHS ingestion when raw survey files exist."""

    dhs_root = Path("data/raw/dhs")
    if dhs_root.exists() and any(dhs_root.iterdir()):
        return _build_multiyear_dhs_features(
            lga_path=lga_path,
            facilities_path=facilities_path,
            population_path=population_path,
            output_path=output_path,
            report_path=report_path,
            coverage_km=coverage_km,
            lga_col=lga_col,
            state_col=state_col,
            pop_lga_col=pop_lga_col,
            pop_state_col=pop_state_col,
            opencellid_path=opencellid_path,
            mock_clusters_path=clusters_path,
        )

    return _build_single_source_features(
        clusters_path=clusters_path,
        lga_path=lga_path,
        facilities_path=facilities_path,
        population_path=population_path,
        output_path=output_path,
        report_path=report_path,
        coverage_km=coverage_km,
        lga_col=lga_col,
        state_col=state_col,
        pop_lga_col=pop_lga_col,
        pop_state_col=pop_state_col,
        opencellid_path=opencellid_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LGA-level features.")
    parser.add_argument("--clusters", type=Path, default=Path("data/raw/mock_dhs_clusters.csv"))
    parser.add_argument("--lga", type=Path, default=Path("data/raw/lga_boundaries.geojson"))
    parser.add_argument("--facilities", type=Path, default=Path("data/raw/health_facilities.geojson"))
    parser.add_argument("--population", type=Path, default=Path("data/processed/population_lga_canonical.csv"))
    parser.add_argument("--opencellid", type=Path, default=Path("data/raw/opencellid.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lga_features.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/build_features_report.json"))
    parser.add_argument("--coverage-km", type=float, default=COVERAGE_KM_DEFAULT)
    parser.add_argument("--lga-col", type=str, default=None, help="Override LGA column name in boundary file.")
    parser.add_argument("--state-col", type=str, default=None, help="Override state column name in boundary file.")
    parser.add_argument("--pop-lga-col", type=str, default=None, help="Override LGA column in population file.")
    parser.add_argument("--pop-state-col", type=str, default=None, help="Override state column in population file.")
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    if not args.lga.exists():
        raise FileNotFoundError(f"Missing LGA boundaries at {args.lga}. Run scripts/download_open_data.py.")
    if not args.facilities.exists():
        raise FileNotFoundError(f"Missing facilities at {args.facilities}. Run scripts/download_open_data.py.")

    dhs_root = Path("data/raw/dhs")
    if not (dhs_root.exists() and any(dhs_root.iterdir())) and not args.clusters.exists():
        raise FileNotFoundError("Missing DHS inputs and mock clusters CSV. Run scripts/create_mock_dhs.py.")

    build_features(
        clusters_path=args.clusters,
        lga_path=args.lga,
        facilities_path=args.facilities,
        population_path=args.population,
        opencellid_path=args.opencellid,
        output_path=args.output,
        report_path=args.report,
        coverage_km=args.coverage_km,
        lga_col=args.lga_col,
        state_col=args.state_col,
        pop_lga_col=args.pop_lga_col,
        pop_state_col=args.pop_state_col,
    )


if __name__ == "__main__":
    main()
