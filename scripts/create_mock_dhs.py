"""Create mock DHS cluster data with spatial structure."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/create_mock_dhs.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_mock_clusters(output_path: Path, n_clusters: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
