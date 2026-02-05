"""Create mock DHS cluster data with spatial structure."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

try:
    from shapely.validation import make_valid
except ImportError:  # shapely < 2.0
    make_valid = None


def configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/create_mock_dhs.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_mock_clusters(output_path: Path, n_clusters: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boundary_path = Path("data/raw/lga_boundaries.geojson")

    if boundary_path.exists():
        lga_gdf = gpd.read_file(boundary_path)
        if lga_gdf.empty:
            logging.warning("Boundary file exists but is empty; falling back to bbox sampling.")
            use_boundary = False
        else:
            if not lga_gdf.geometry.is_valid.all():
                if make_valid:
                    lga_gdf["geometry"] = lga_gdf.geometry.apply(make_valid)
                else:
                    lga_gdf["geometry"] = lga_gdf.geometry.buffer(0)
            union_geom = lga_gdf.unary_union
            if not union_geom.is_valid:
                union_geom = union_geom.buffer(0)
            bounds = lga_gdf.total_bounds  # minx, miny, maxx, maxy
            minx, miny, maxx, maxy = bounds

            accepted_lat = []
            accepted_lon = []
            attempts = 0
            max_attempts = 50 * n_clusters

            while len(accepted_lat) < n_clusters and attempts < max_attempts:
                cand_lon = rng.uniform(minx, maxx)
                cand_lat = rng.uniform(miny, maxy)
                attempts += 1
                if union_geom.intersects(Point(cand_lon, cand_lat)):
                    accepted_lat.append(cand_lat)
                    accepted_lon.append(cand_lon)

            if len(accepted_lat) < n_clusters:
                raise ValueError(
                    "Failed to sample enough points within LGA boundaries. "
                    f"Accepted {len(accepted_lat)} of {n_clusters} after {attempts} attempts. "
                    "Check boundary geometry validity."
                )

            lat = np.array(accepted_lat)
            lon = np.array(accepted_lon)
            acceptance_rate = len(accepted_lat) / attempts
            sample_bounds = (float(min(lon)), float(max(lon)), float(min(lat)), float(max(lat)))
            logging.info(
                "Sampled mock clusters within LGA union: acceptance=%.2f%% over %d attempts; "
                "sample bounds lon=[%.3f, %.3f], lat=[%.3f, %.3f]",
                acceptance_rate * 100,
                attempts,
                sample_bounds[0],
                sample_bounds[1],
                sample_bounds[2],
                sample_bounds[3],
            )
            print(
                f"Sampling within LGA union: acceptance={acceptance_rate:.2%} over {attempts} attempts; "
                f"sample bounds lon=[{sample_bounds[0]:.3f},{sample_bounds[1]:.3f}], "
                f"lat=[{sample_bounds[2]:.3f},{sample_bounds[3]:.3f}]"
            )
            use_boundary = True
    else:
        use_boundary = False

    if not use_boundary:
        lat = rng.uniform(4.0, 14.2, size=n_clusters)
        lon = rng.uniform(2.5, 14.8, size=n_clusters)

    hub_centers = np.array(
        [
            [6.5, 3.4],
            [9.1, 7.4],
            [12.0, 8.5],
            [5.0, 7.5],
            [10.5, 3.0],
        ]
    )
    coords = np.column_stack([lat, lon])
    distances = np.linalg.norm(coords[:, None, :] - hub_centers[None, :, :], axis=2)
    min_distance = distances.min(axis=1)

    urban_prob = np.clip(1.2 - min_distance / 3.5, 0.1, 0.9)
    urban = rng.binomial(1, urban_prob)

    access_score = np.clip(1.5 - min_distance / 4.0 + urban * 0.2, 0.1, 1.5)
    baseline_u5mr = 85 - access_score * 20 + rng.normal(0, 8, size=n_clusters)
    baseline_u5mr = np.clip(baseline_u5mr, 25, 140)

    live_births = rng.poisson(150 + urban * 40, size=n_clusters)
    u5_deaths = rng.binomial(live_births, baseline_u5mr / 1000.0)
    u5mr = (u5_deaths / live_births) * 1000.0

    df = pd.DataFrame(
        {
            "cluster_id": np.arange(1, n_clusters + 1),
            "lat": lat,
            "lon": lon,
            "urban": urban,
            "live_births": live_births,
            "u5_deaths": u5_deaths,
            "u5mr": u5mr,
            "access_score": access_score,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Created %d mock clusters at %s", n_clusters, output_path)
    print(
        f"OK: {len(df)} clusters | u5mr mean={df['u5mr'].mean():.2f} | "
        f"urban share={df['urban'].mean():.2f}"
    )
    return df


def main() -> None:
    configure_logging()
    output_path = Path("data/raw/mock_dhs_clusters.csv")
    create_mock_clusters(output_path)


if __name__ == "__main__":
    main()
