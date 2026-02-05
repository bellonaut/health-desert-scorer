"""Aggregate population per LGA from WorldPop raster for Stage A."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src.data.spatial_ops import normalize_admin_name


def _detect_column(gdf: gpd.GeoDataFrame, candidates: list[str], override: str | None) -> str:
    if override and override in gdf.columns:
        return override
    lower = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"None of columns {candidates} found in boundary file. Columns present: {list(gdf.columns)}")


def aggregate_population(
    lga_path: Path,
    raster_path: Path,
    output_path: Path,
    lga_col: str | None = None,
    state_col: str | None = None,
) -> pd.DataFrame:
    logging.info("Reading boundaries from %s", lga_path)
    lgas = gpd.read_file(lga_path)

    lga_col = _detect_column(
        lgas,
        ["lganame", "lga_name", "NAME_2", "lga", "name"],
        lga_col,
    )
    state_col = _detect_column(
        lgas,
        ["statename", "state_name", "NAME_1", "state"],
        state_col,
    )

    lgas = lgas.rename(columns={lga_col: "lga_name", state_col: "state_name"})

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        logging.info("Raster CRS: %s | dtype: %s | nodata: %s", raster_crs, src.dtypes[0], nodata)

    if lgas.crs != raster_crs:
        logging.info("Reprojecting LGAs from %s to %s", lgas.crs, raster_crs)
        lgas = lgas.to_crs(raster_crs)

    logging.info("Running zonal_stats (sum)...")
    zs = zonal_stats(
        lgas,
        raster_path,
        stats=["sum"],
        nodata=nodata,
        all_touched=True,
    )

    pop = []
    for z in zs:
        val = z.get("sum")
        if val is None or np.isnan(val):
            val = 0.0
        pop.append(float(val))

    lgas["population"] = pop
    lgas["population"] = lgas["population"].clip(lower=0).round().astype("int64")
    lgas["lga_name"] = lgas["lga_name"].astype(str)
    lgas["state_name"] = lgas["state_name"].astype(str)

    out = lgas[["lga_name", "state_name", "population"]].copy()

    # Normalize for downstream joins while keeping raw strings
    out["state_lga_norm"] = normalize_admin_name(out["state_name"]) + "__" + normalize_admin_name(out["lga_name"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out[["lga_name", "state_name", "population"]].to_csv(output_path, index=False)

    desc = out["population"].describe()
    logging.info(
        "Population stats: count=%d min=%d median=%.0f max=%d",
        int(desc["count"]),
        int(desc["min"]),
        desc["50%"],
        int(desc["max"]),
    )
    assert (out["population"] >= 0).all()
    print(
        f"Saved {len(out)} rows to {output_path} | min={desc['min']:.0f} median={desc['50%']:.0f} max={desc['max']:.0f}"
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate population per LGA from raster.")
    parser.add_argument("--lga", type=Path, default=Path("data/raw/lga_boundaries.geojson"))
    parser.add_argument("--raster", type=Path, default=Path("data/raw/nga_ppp_2020_constrained.tif"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/population_lga.csv"))
    parser.add_argument("--lga-col", type=str, default=None, help="Override LGA column name in boundaries.")
    parser.add_argument("--state-col", type=str, default=None, help="Override state column name in boundaries.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.lga.exists():
        raise FileNotFoundError(f"Boundary file not found: {args.lga}")
    if not args.raster.exists():
        raise FileNotFoundError(f"Raster file not found: {args.raster}")
    aggregate_population(
        lga_path=args.lga,
        raster_path=args.raster,
        output_path=args.output,
        lga_col=args.lga_col,
        state_col=args.state_col,
    )


if __name__ == "__main__":
    main()
