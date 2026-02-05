"""Build LGA-level features from DHS clusters and facility data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.spatial_ops import (
    CRS,
    aggregate_facility_metrics_by_lga,
    assign_points_to_lga,
    coverage_within_km,
    aggregate_tower_metrics_by_lga,
    infer_lga_names_from_facilities,
    load_opencellid,
    load_facilities,
    load_lga_boundaries,
    make_points_from_latlon,
    normalize_admin_name,
)


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
    df["lat"] = df["latitude"]
    df["lon"] = df["longitude"]

    if urban_src:
        df = df.rename(columns={urban_src: "urban"}) if urban_src != "urban" else df
        urban_series = df["urban"]
        if urban_series.dtype == bool:
            df["urban"] = urban_series.astype(int)
        elif pd.api.types.is_numeric_dtype(urban_series):
            # Keep as numeric 0/1
            df["urban"] = urban_series.astype(float)
        else:
            mapped = (
                urban_series.astype(str)
                .str.strip()
                .str.upper()
                .map({"URBAN": 1, "U": 1, "RURAL": 0, "R": 0})
            )
            df["urban"] = mapped
            # If mapping failed (NaN), leave as NaN to avoid misleading mean
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


def _validate_features(df: pd.DataFrame) -> None:
    if df["lga_uid"].duplicated().any():
        dup_keys = df.loc[df["lga_uid"].duplicated(), "lga_uid"].head(10).tolist()
        raise ValueError(f"Duplicate lga_uid entries found in features. Sample duplicates: {dup_keys}")
    dup_lga_names = df["lga_name"].value_counts()
    dup_lga_names = dup_lga_names[dup_lga_names > 1]
    if not dup_lga_names.empty:
        logging.info("Found %d duplicated lga_name values; top: %s", len(dup_lga_names), dup_lga_names.head(10).to_dict())
    if len(df) < 500:
        logging.warning("Row count (%d) below expected Nigeria LGA count.", len(df))
    if not df["avg_distance_km"].between(0, 200).all():
        raise ValueError("avg_distance_km values out of expected range 0-200 km.")
    if (df["facilities_per_10k"] < 0).any():
        raise ValueError("facilities_per_10k contains negative values.")
    labeled = df["u5mr_mean"].notna().sum()
    logging.info("Label coverage: %d of %d LGAs (%.1f%%) have u5mr_mean", labeled, len(df), labeled / len(df) * 100 if len(df) else 0.0)


def _build_report(
    df: pd.DataFrame,
    output_path: Path,
    tower_rows_loaded: int = 0,
    tower_lga_coverage: float = 0.0,
) -> None:
    report = {
        "rows": len(df),
        "avg_distance_km_mean": float(df["avg_distance_km"].mean()),
        "u5mr_mean_mean": float(df["u5mr_mean"].mean()),
        "facilities_count_total": int(df["facilities_count"].sum()),
        "population_lgas_with_values": int(df["population"].notna().sum()),
        "tower_rows_loaded": int(tower_rows_loaded),
        "tower_lga_coverage_pct": float(tower_lga_coverage),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))


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
    logging.info("Loading DHS clusters from %s", clusters_path)
    clusters = _normalize_clusters(pd.read_csv(clusters_path))
    points = make_points_from_latlon(clusters, lat_col="latitude", lon_col="longitude")

    lgas = load_lga_boundaries(str(lga_path), lga_col=lga_col, state_col=state_col)
    facilities = load_facilities(str(facilities_path))

    need_inference = (
        "lga_name" not in lgas.columns
        or lgas["lga_name"].isna().all()
        or lgas["lga_name"].astype(str).str.strip().eq("").all()
    )
    if need_inference or lgas["lga_name"].isna().any() or lgas["lga_name"].astype(str).str.strip().eq("").any():
        logging.info("Inferring LGA names via facilities overlay.")
        lgas = infer_lga_names_from_facilities(lgas, facilities)
        non_placeholder_fraction = (
            ~lgas["lga_name"].astype(str).str.startswith("LGA_")
        ).mean()
        if non_placeholder_fraction < 0.5:
            raise ValueError(
                "Fewer than 50% of polygons received non-placeholder LGA names after inference; "
                "the boundary file may not match the facilities geography."
            )

    # Identity and composite keys
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

    joined = assign_points_to_lga(points, lgas)
    joined["u5mr"] = joined["u5_deaths"] / joined["live_births"] * 1000.0
    joined["lga_uid"] = joined["lga_id"]

    def _first_nonnull(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    outcomes = joined.groupby("lga_uid").agg(
        u5mr_mean=("u5mr", "mean"),
        u5mr_median=("u5mr", "median"),
        live_births_sum=("live_births", "sum"),
        u5_deaths_sum=("u5_deaths", "sum"),
        urban_prop=("urban", "mean"),
        lga_lat=("lat", "mean"),
        lga_lon=("lon", "mean"),
    ).reset_index()

    # Base table: all LGAs
    base = lgas[["lga_uid", "state_name", "lga_name", "state_lga", "state_lga_norm"]].drop_duplicates("lga_uid")

    grouped = base.merge(outcomes, on="lga_uid", how="left")

    population_df = None
    if population_path and population_path.exists():
        population_df = pd.read_csv(population_path)
        expected = {"lga_name", "state_name", "population"}
        missing_cols = expected - set(population_df.columns)
        if missing_cols:
            raise ValueError(f"Population file missing required columns: {missing_cols}")
        population_df["lga_name"] = population_df["lga_name"].astype(str).fillna("")
        population_df["state_name"] = population_df["state_name"].astype(str).fillna("")
        population_df["state_lga_norm"] = _norm_key(population_df["state_name"]) + "__" + _norm_key(population_df["lga_name"])

    facilities_metrics = aggregate_facility_metrics_by_lga(facilities, lgas, population_df=None)
    facilities_metrics = facilities_metrics.rename(
        columns={"avg_distance_km_proxy": "avg_distance_km", "lga_id": "lga_uid"}
    )
    facilities_metrics = facilities_metrics[
        ["lga_uid", "avg_distance_km", "facilities_count", "facilities_per_10k"]
    ]

    tower_metrics = None
    tower_rows_loaded = 0
    tower_lga_coverage = 0.0
    if opencellid_path and Path(opencellid_path).exists():
        towers = load_opencellid(str(opencellid_path))
        tower_rows_loaded = len(towers)
        tower_metrics = aggregate_tower_metrics_by_lga(towers, lgas)
        tower_lga_coverage = (tower_metrics["towers_count"] > 0).mean() * 100 if len(tower_metrics) else 0.0

    coverage_df = coverage_within_km(lgas, facilities, km=coverage_km)
    if "population_covered_pct" in coverage_df.columns:
        coverage_df = coverage_df.rename(columns={"population_covered_pct": "coverage_5km"})
    else:
        coverage_df = coverage_df.rename(columns={"area_covered_pct": "coverage_5km"})
    coverage_df = coverage_df.rename(columns={"lga_id": "lga_uid"})

    features = grouped.merge(facilities_metrics, on="lga_uid", how="left").merge(
        coverage_df[["lga_uid", "coverage_5km"]], on="lga_uid", how="left"
    )
    if tower_metrics is not None:
        features = features.merge(tower_metrics, on="lga_uid", how="left")

    # Area in km^2 for density
    lga_area = lgas[["lga_uid", "geometry"]].copy()
    lga_area = lga_area.to_crs(CRS.metric)
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
            "Population merge: matched %d / %d LGAs (%.1f%%). Sample unmatched keys: %s",
            matched,
            len(features),
            coverage * 100,
            unmatched_keys,
        )
        min_pop = features["population"].min(skipna=True)
        max_pop = features["population"].max(skipna=True)
        zero_pop = ((features["population"] == 0) & features["population"].notna()).sum()
        logging.info(
            "Population stats: min=%.0f max=%.0f zero_pop_lgas=%d",
            min_pop if pd.notna(min_pop) else float("nan"),
            max_pop if pd.notna(max_pop) else float("nan"),
            zero_pop,
        )
        if coverage < 0.98:
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
        "towers_count",
        "tower_density_per_km2",
        "avg_dist_to_tower_km",
        "towers_per_10k_pop",
        "coverage_5km",
    ]
    for col in ordered_cols:
        if col not in features.columns:
            features[col] = np.nan
    features = features[ordered_cols]

    _validate_features(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    _build_report(
        features,
        report_path,
        tower_rows_loaded=tower_rows_loaded,
        tower_lga_coverage=tower_lga_coverage,
    )
    logging.info(
        "OK: built features with %d LGAs | towers_mean=%.2f | pct_LGAs_with_tower=%.1f%%",
        len(features),
        features["towers_count"].mean() if "towers_count" in features else 0,
        (features["towers_count"] > 0).mean() * 100 if "towers_count" in features else 0,
    )
    print(
        f"OK: {len(features)} LGAs | avg_distance_km mean={features['avg_distance_km'].mean():.2f} | "
        f"u5mr_mean mean={features['u5mr_mean'].mean():.2f} | "
        f"towers_count mean={features['towers_count'].mean() if 'towers_count' in features else 0:.2f}"
    )
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LGA-level features.")
    parser.add_argument("--clusters", type=Path, default=Path("data/raw/mock_dhs_clusters.csv"))
    parser.add_argument("--lga", type=Path, default=Path("data/raw/lga_boundaries.geojson"))
    parser.add_argument("--facilities", type=Path, default=Path("data/raw/health_facilities.geojson"))
    parser.add_argument("--population", type=Path, default=Path("data/processed/population_lga_canonical.csv"))
    parser.add_argument("--opencellid", type=Path, default=Path("data/raw/opencellid.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lga_features.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/build_features_report.json"))
    parser.add_argument("--coverage-km", type=float, default=5.0)
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
    if not args.clusters.exists():
        raise FileNotFoundError("Missing clusters CSV. Run scripts/create_mock_dhs.py.")

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
