"""Spatial operations and CRS utilities for Nigeria health desert scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRSConfig:
    """Centralized CRS configuration."""

    wgs84: str = "EPSG:4326"
    metric: str = "EPSG:3857"


CRS = CRSConfig()


def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame missing CRS; please set a CRS before operations.")
    if gdf.crs.to_string() != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _normalize_columns(
    gdf: gpd.GeoDataFrame,
    name_candidates: Iterable[str],
    state_candidates: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    lower_map = {col.lower(): col for col in gdf.columns}
    name_col = next((lower_map.get(c.lower()) for c in name_candidates if c.lower() in lower_map), None)
    if not name_col:
        raise ValueError(f"Could not find LGA name column among {name_candidates}.")
    gdf = gdf.rename(columns={name_col: "lga_name"})
    if state_candidates:
        state_col = next(
            (lower_map.get(c.lower()) for c in state_candidates if c.lower() in lower_map),
            None,
        )
        if state_col:
            gdf = gdf.rename(columns={state_col: "state_name"})
    return gdf


def load_lga_boundaries(path: str) -> gpd.GeoDataFrame:
    """Load LGA boundaries and normalize column names."""

    gdf = gpd.read_file(path)
    gdf = _normalize_columns(
        gdf,
        name_candidates=("lga_name", "lga", "lga_name_en", "name", "lga_nam"),
        state_candidates=("state_name", "state", "admin1", "state_nam"),
    )
    gdf = gdf[["lga_name", "state_name", "geometry"]].copy() if "state_name" in gdf else gdf[
        ["lga_name", "geometry"]
    ].copy()
    gdf["lga_name"] = gdf["lga_name"].astype(str).str.strip()
    if "state_name" in gdf:
        gdf["state_name"] = gdf["state_name"].astype(str).str.strip()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.wgs84)
    return gdf


def load_facilities(path: str) -> gpd.GeoDataFrame:
    """Load health facility points and normalize columns."""

    gdf = gpd.read_file(path)
    lower_map = {col.lower(): col for col in gdf.columns}
    name_col = lower_map.get("facility_name") or lower_map.get("name")
    type_col = lower_map.get("facility_type") or lower_map.get("type")
    if name_col:
        gdf = gdf.rename(columns={name_col: "facility_name"})
    else:
        gdf["facility_name"] = "unknown"
    if type_col:
        gdf = gdf.rename(columns={type_col: "facility_type"})
    else:
        gdf["facility_type"] = "unknown"
    gdf = gdf[["facility_name", "facility_type", "geometry"]].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.wgs84)
    return gdf


def assign_points_to_lga(points_gdf: gpd.GeoDataFrame, lga_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatially join points to LGAs with validation."""

    points = _ensure_crs(points_gdf, CRS.wgs84)
    lgas = _ensure_crs(lga_gdf, CRS.wgs84)
    joined = gpd.sjoin(points, lgas[["lga_name", "state_name", "geometry"]] if "state_name" in lgas else lgas,
                       how="left", predicate="within")
    join_rate = joined["lga_name"].notna().mean()
    if join_rate < 0.7:
        raise ValueError(f"Spatial join success rate too low: {join_rate:.2%}")
    LOGGER.info("Spatial join success rate: %.2f%%", join_rate * 100)
    return joined


def compute_nearest_facility_distance(
    points_gdf: gpd.GeoDataFrame, facilities_gdf: gpd.GeoDataFrame
) -> pd.Series:
    """Compute nearest facility distance in kilometers using projected CRS."""

    points = _ensure_crs(points_gdf, CRS.wgs84).to_crs(CRS.metric)
    facilities = _ensure_crs(facilities_gdf, CRS.wgs84).to_crs(CRS.metric)
    if points.empty or facilities.empty:
        raise ValueError("Points and facilities must be non-empty to compute distances.")
    joined = gpd.sjoin_nearest(points, facilities, how="left", distance_col="distance_m")
    return joined["distance_m"].astype(float) / 1000.0


def aggregate_facility_metrics_by_lga(
    facilities_gdf: gpd.GeoDataFrame,
    lga_gdf: gpd.GeoDataFrame,
    population_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate facility metrics by LGA."""

    facilities = _ensure_crs(facilities_gdf, CRS.wgs84)
    lgas = _ensure_crs(lga_gdf, CRS.wgs84)
    facilities_joined = gpd.sjoin(facilities, lgas[["lga_name", "geometry"]], how="left", predicate="within")
    counts = facilities_joined.groupby("lga_name").size().rename("facilities_count").reset_index()
    lga_centroids = lgas.copy()
    lga_centroids["geometry"] = lga_centroids.geometry.centroid
    centroid_distances = compute_nearest_facility_distance(lga_centroids, facilities)
    lga_centroids["avg_distance_km_proxy"] = centroid_distances.values
    metrics = lga_centroids[["lga_name", "avg_distance_km_proxy"]].merge(counts, on="lga_name", how="left")
    metrics["facilities_count"] = metrics["facilities_count"].fillna(0).astype(int)
    if population_df is not None and "population" in population_df.columns:
        pop = population_df[["lga_name", "population"]].copy()
        metrics = metrics.merge(pop, on="lga_name", how="left")
        metrics["facilities_per_10k"] = metrics["facilities_count"] / (metrics["population"] / 10000.0)
    else:
        metrics["facilities_per_10k"] = np.nan
    return metrics


def coverage_within_km(
    lga_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    km: float = 5.0,
    population_raster: Optional[str] = None,
) -> pd.DataFrame:
    """Estimate coverage within km. Uses area coverage if population raster is missing."""

    lgas = _ensure_crs(lga_gdf, CRS.wgs84).to_crs(CRS.metric)
    facilities = _ensure_crs(facilities_gdf, CRS.wgs84).to_crs(CRS.metric)
    buffer_geom = facilities.buffer(km * 1000.0)
    coverage = []

    if population_raster:
        import importlib.util

        if importlib.util.find_spec("rasterio") is None:
            LOGGER.warning("Rasterio not available; falling back to area-based coverage.")
            population_raster = None
        else:
            import rasterio
            from rasterio.mask import mask

            with rasterio.open(population_raster) as src:
                for _, row in lgas.iterrows():
                    geom = [row.geometry]
                    total, _ = mask(src, geom, crop=True)
                    covered, _ = mask(src, buffer_geom.intersection(row.geometry), crop=True)
                    total_pop = np.nansum(total)
                    covered_pop = np.nansum(covered)
                    pct = (covered_pop / total_pop) * 100 if total_pop > 0 else np.nan
                    coverage.append({"lga_name": row["lga_name"], "population_covered_pct": pct})
            return pd.DataFrame(coverage)

    for _, row in lgas.iterrows():
        lga_area = row.geometry.area
        covered_area = buffer_geom.intersection(row.geometry).area.sum()
        pct = (covered_area / lga_area) * 100 if lga_area > 0 else np.nan
        coverage.append({"lga_name": row["lga_name"], "area_covered_pct": pct})
    return pd.DataFrame(coverage)


def make_points_from_latlon(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> gpd.GeoDataFrame:
    """Create GeoDataFrame from lat/lon."""

    if lat_col not in df or lon_col not in df:
        raise ValueError(f"Missing lat/lon columns: {lat_col}, {lon_col}.")
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=CRS.wgs84)
