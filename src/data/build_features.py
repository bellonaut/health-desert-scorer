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
    load_facilities,
    load_lga_boundaries,
    make_points_from_latlon,
)


def _configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/build_features.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _validate_features(df: pd.DataFrame) -> None:
    if df["lga_name"].duplicated().any():
        raise ValueError("Duplicate lga_name entries found in features.")
    if len(df) < 500:
        logging.warning("Row count (%d) below expected Nigeria LGA count.", len(df))
    if not df["avg_distance_km"].between(0, 200).all():
        raise ValueError("avg_distance_km values out of expected range 0-200 km.")
    if (df["facilities_per_10k"] < 0).any():
        raise ValueError("facilities_per_10k contains negative values.")


def _build_report(df: pd.DataFrame, output_path: Path) -> None:
    report = {
        "rows": len(df),
        "avg_distance_km_mean": float(df["avg_distance_km"].mean()),
        "u5mr_mean_mean": float(df["u5mr_mean"].mean()),
        "facilities_count_total": int(df["facilities_count"].sum()),
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
) -> pd.DataFrame:
    logging.info("Loading DHS clusters from %s", clusters_path)
    clusters = pd.read_csv(clusters_path)
    points = make_points_from_latlon(clusters, lat_col="lat", lon_col="lon")

    lgas = load_lga_boundaries(str(lga_path))
    facilities = load_facilities(str(facilities_path))

    joined = assign_points_to_lga(points, lgas)
    joined["u5mr"] = joined["u5_deaths"] / joined["live_births"] * 1000.0

    grouped = joined.groupby("lga_name").agg(
        u5mr_mean=("u5mr", "mean"),
        u5mr_median=("u5mr", "median"),
        live_births_sum=("live_births", "sum"),
        u5_deaths_sum=("u5_deaths", "sum"),
        urban_prop=("urban", "mean"),
        lga_lat=("lat", "mean"),
        lga_lon=("lon", "mean"),
    )
    grouped = grouped.reset_index()

    if "state_name" in joined:
        state_map = joined.groupby("lga_name")["state_name"].agg(lambda x: x.dropna().iloc[0] if len(x) else None)
        grouped = grouped.merge(state_map.rename("state_name"), on="lga_name", how="left")

    population_df = None
    if population_path and population_path.exists():
        population_df = pd.read_csv(population_path)
        if "population" not in population_df.columns:
            raise ValueError("Population file must include 'population' column.")

    facilities_metrics = aggregate_facility_metrics_by_lga(facilities, lgas, population_df)
    facilities_metrics = facilities_metrics.rename(columns={"avg_distance_km_proxy": "avg_distance_km"})

    coverage_df = coverage_within_km(lgas, facilities, km=coverage_km)
    if "population_covered_pct" in coverage_df.columns:
        coverage_df = coverage_df.rename(columns={"population_covered_pct": "coverage_5km"})
    else:
        coverage_df = coverage_df.rename(columns={"area_covered_pct": "coverage_5km"})

    features = grouped.merge(facilities_metrics, on="lga_name", how="left").merge(
        coverage_df, on="lga_name", how="left"
    )

    if population_df is not None and "population" in population_df.columns:
        features = features.merge(population_df[["lga_name", "population"]], on="lga_name", how="left")
        if "area_sq_km" in population_df.columns:
            features["population_density"] = features["population"] / population_df["area_sq_km"]
    else:
        features["population"] = np.nan
        features["population_density"] = np.nan

    ordered_cols = [
        "lga_name",
        "state_name",
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
        "coverage_5km",
    ]
    for col in ordered_cols:
        if col not in features.columns:
            features[col] = np.nan
    features = features[ordered_cols]

    _validate_features(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    _build_report(features, report_path)
    logging.info("OK: built features with %d LGAs", len(features))
    print(
        f"OK: {len(features)} LGAs | avg_distance_km mean={features['avg_distance_km'].mean():.2f} | "
        f"u5mr_mean mean={features['u5mr_mean'].mean():.2f}"
    )
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LGA-level features.")
    parser.add_argument("--clusters", type=Path, default=Path("data/raw/mock_dhs_clusters.csv"))
    parser.add_argument("--lga", type=Path, default=Path("data/raw/lga_boundaries.geojson"))
    parser.add_argument("--facilities", type=Path, default=Path("data/raw/health_facilities.geojson"))
    parser.add_argument("--population", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed/lga_features.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/build_features_report.json"))
    parser.add_argument("--coverage-km", type=float, default=5.0)
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
        output_path=args.output,
        report_path=args.report,
        coverage_km=args.coverage_km,
    )


if __name__ == "__main__":
    main()
