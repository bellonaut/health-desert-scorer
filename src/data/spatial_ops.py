"""Spatial operations and CRS utilities for Nigeria health desert scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRSConfig:
    """Centralized CRS configuration."""

    wgs84: str = "EPSG:4326"
    metric: str = "EPSG:3857"


CRS = CRSConfig()


def normalize_admin_name(series: pd.Series) -> pd.Series:
    """Normalize admin names for robust joining (uppercase, trim, collapse whitespace, remove punctuation)."""

    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[./\\'-]", " ", regex=True)
        .str.replace(r"\\s+", " ", regex=True)
        .str.strip()
    )


def load_opencellid(path: str) -> gpd.GeoDataFrame:
    """
    Load OpenCellID CSV (gzip) and return GeoDataFrame of towers.

    Expected columns (no header): radio,mcc,net,area,cell,unit,lon,lat,range,samples,changeable,created,updated,averageSignal
    """

    col_names = [
        "radio",
        "mcc",
        "net",
        "area",
        "cell",
        "unit",
        "lon",
        "lat",
        "range",
        "samples",
        "changeable",
        "created",
        "updated",
        "averageSignal",
    ]
    df = pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=col_names,
        dtype={
            "radio": "string",
            "mcc": "Int64",
            "net": "Int64",
            "area": "Int64",
            "cell": "Int64",
            "unit": "Int64",
            "lon": "float64",
            "lat": "float64",
            "range": "float64",
            "samples": "Int64",
            "changeable": "Int64",
            "created": "Int64",
            "updated": "Int64",
            "averageSignal": "float64",
        },
    )
    total_rows = len(df)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])
    kept_rows = len(df)
    LOGGER.info("Loaded OpenCellID: %d rows (kept %d with valid coords)", total_rows, kept_rows)
    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.wgs84)
    return gdf


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
    source_path: str | None = None,
) -> gpd.GeoDataFrame:
    """Rename LGA/state columns to standardized names."""

    lower_map = {col.lower(): col for col in gdf.columns}
    name_candidates = [c for c in name_candidates if c]
    name_col = next((lower_map.get(c.lower()) for c in name_candidates if c.lower() in lower_map), None)
    if not name_col:
        cols_preview = list(gdf.columns)[:30]
        location = f" in {source_path}" if source_path else ""
        raise ValueError(
            f"Could not find LGA name column{location}. Tried candidates: {name_candidates}. "
            f"Columns present (first 30): {cols_preview}. "
            "Pass --lga-col if using build_features CLI."
        )
    gdf = gdf.rename(columns={name_col: "lga_name"})
    if state_candidates:
        state_col = next(
            (lower_map.get(c.lower()) for c in state_candidates if c.lower() in lower_map),
            None,
        )
        if state_col:
            gdf = gdf.rename(columns={state_col: "state_name"})
    return gdf


def load_lga_boundaries(
    path: str,
    lga_col: str | None = None,
    state_col: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Load LGA boundaries and normalize column names.

    If no LGA name column can be located, returns the raw GeoDataFrame with an
    empty ``lga_name`` column so downstream inference can populate it.
    """

    gdf = gpd.read_file(path)
    name_candidates = ([lga_col] if lga_col else []) + [
        "lga_name",
        "lga",
        "lga_name_en",
        "name",
        "lga_nam",
        "adm2_name",
        "ADM2_NAME",
        "adm2",
        "ADM2",
        "admin2",
        "ADMIN2",
        "admin_2",
        "NAME_2",
        "NAME2",
        "shapeName",
        "shapename",
        "district",
        "county",
        "area",
        "lg_name",
        "lga_nam_e",
        "lgaName",
    ]
    state_candidates = ([state_col] if state_col else []) + [
        "state_name",
        "state",
        "admin1",
        "ADMIN1",
        "adm1_name",
        "ADM1_NAME",
        "NAME_1",
        "STATE",
        "statename",
        "state_nam",
        "STATE_NAME",
    ]

    try:
        gdf = _normalize_columns(
            gdf,
            name_candidates=name_candidates,
            state_candidates=state_candidates,
            source_path=path,
        )
        gdf = (
            gdf[["lga_name", "state_name", "geometry"]].copy()
            if "state_name" in gdf
            else gdf[["lga_name", "geometry"]].copy()
        )
        gdf["lga_name"] = gdf["lga_name"].astype(str).str.strip()
        if "state_name" in gdf:
            gdf["state_name"] = gdf["state_name"].astype(str).str.strip()
        LOGGER.info("LGA names read directly from boundary properties.")
    except ValueError as exc:
        LOGGER.warning("Boundary file missing LGA names; will attempt inference later. %s", exc)
        gdf = gdf.copy()
        if "lga_name" not in gdf.columns:
            gdf["lga_name"] = pd.NA
        if "state_name" not in gdf.columns and state_col and state_col in gdf.columns:
            gdf = gdf.rename(columns={state_col: "state_name"})

    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.wgs84)
    gdf = gdf[gdf.geometry.notna()]
    if "lga_id" not in gdf.columns:
        gdf["lga_id"] = np.arange(len(gdf))
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
    lga_col = next((lower_map.get(c) for c in ("lga", "lga_name") if c in lower_map), None)
    state_col = next((lower_map.get(c) for c in ("state", "state_name", "statename") if c in lower_map), None)
    if lga_col and lga_col != "lga":
        gdf = gdf.rename(columns={lga_col: "lga"})
    if state_col and state_col != "state":
        gdf = gdf.rename(columns={state_col: "state"})
    cols = ["facility_name", "facility_type", "geometry"]
    if "lga" in gdf.columns:
        cols.append("lga")
    if "state" in gdf.columns:
        cols.append("state")
    gdf = gdf[cols].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.wgs84)
    return gdf


def _deterministic_mode(series: pd.Series) -> str | None:
    """Return deterministic mode (alphabetical tie-break) from a series."""

    if series is None:
        return None
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return None
    counts = cleaned.value_counts()
    max_count = counts.max()
    top = sorted(counts[counts == max_count].index)
    return top[0] if top else None


def infer_lga_names_from_facilities(
    lga_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    facility_lga_col: str = "lga",
) -> gpd.GeoDataFrame:
    """
    Infer LGA (and optionally state) names by spatially joining facilities to polygons.

    Uses majority vote per polygon; fills missing with deterministic placeholders.
    """

    lgas = _ensure_crs(lga_gdf, CRS.wgs84).copy()
    facilities = _ensure_crs(facilities_gdf, CRS.wgs84).copy()

    if facility_lga_col not in facilities.columns:
        raise ValueError(f"Facilities missing '{facility_lga_col}' column required to infer LGA names.")

    if not lgas.geometry.is_valid.all():
        lgas["geometry"] = lgas.geometry.buffer(0)

    join_cols = [facility_lga_col, "geometry"]
    if "state" in facilities.columns:
        join_cols.append("state")

    joined = gpd.sjoin(
        facilities[join_cols],
        lgas[["geometry"]],
        how="left",
        predicate="within",
    ).rename(columns={"index_right": "lga_index"})

    name_map = {}
    state_map = {}
    for idx, group in joined.groupby("lga_index"):
        if pd.isna(idx):
            continue
        mode_name = _deterministic_mode(group[facility_lga_col])
        if mode_name:
            name_map[idx] = mode_name
        if "state" in group.columns:
            mode_state = _deterministic_mode(group["state"])
            if mode_state:
                state_map[idx] = mode_state

    if "lga_name" not in lgas.columns:
        lgas["lga_name"] = pd.NA

    existing_valid = lgas["lga_name"].notna() & lgas["lga_name"].astype(str).str.strip().ne("")
    inferred_names = lgas.index.to_series().map(name_map.get)
    lgas.loc[~existing_valid, "lga_name"] = inferred_names.loc[~existing_valid]

    placeholder_mask = lgas["lga_name"].isna() | lgas["lga_name"].astype(str).str.strip().eq("")
    lgas.loc[placeholder_mask, "lga_name"] = [
        f"LGA_{idx}" for idx in lgas.index[placeholder_mask]
    ]

    if state_map:
        inferred_states = lgas.index.to_series().map(state_map.get)
        if "state_name" not in lgas.columns:
            lgas["state_name"] = inferred_states
        else:
            state_valid = lgas["state_name"].notna() & lgas["state_name"].astype(str).str.strip().ne("")
            lgas.loc[~state_valid, "state_name"] = inferred_states.loc[~state_valid]

    total = len(lgas)
    mode_assigned = inferred_names.loc[~existing_valid].notna().sum()
    placeholders = lgas["lga_name"].astype(str).str.startswith("LGA_").sum()
    mode_pct = (mode_assigned / total * 100) if total else 0.0
    placeholder_pct = (placeholders / total * 100) if total else 0.0
    LOGGER.info(
        "LGA names inferred from facilities by spatial overlay: %.1f%% via mode, %.1f%% placeholders.",
        mode_pct,
        placeholder_pct,
    )
    return lgas


def assign_points_to_lga(points_gdf: gpd.GeoDataFrame, lga_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatially join points to LGAs with validation and recovery attempts."""

    if lga_gdf.crs is None:
        raise ValueError("LGA GeoDataFrame missing CRS; cannot perform spatial join.")
    if points_gdf.crs is None:
        raise ValueError("Points GeoDataFrame missing CRS; cannot perform spatial join.")

    orig_points = points_gdf.copy()
    lgas = lga_gdf
    points = points_gdf.copy()
    if points.crs != lgas.crs:
        points = points.to_crs(lgas.crs)

    min_success_rate = 0.7
    attempts = []

    def _attempt_join(p_gdf: gpd.GeoDataFrame, predicate: str, label: str):
        join_cols = ["lga_name", "geometry", "lga_id"]
        if "state_name" in lgas:
            join_cols.insert(1, "state_name")
        joined_local = gpd.sjoin(p_gdf, lgas[join_cols], how="left", predicate=predicate)
        rate_local = joined_local["lga_name"].notna().mean()
        attempts.append((label, rate_local, joined_local))
        return joined_local, rate_local

    joined, rate = _attempt_join(points, "within", "within")
    if rate < min_success_rate:
        LOGGER.warning("Spatial join success rate low with predicate=within: %.2f%%; retrying with intersects.",
                       rate * 100)
        joined, rate = _attempt_join(points, "intersects", "intersects")

    if rate < min_success_rate:
        LOGGER.warning("Spatial join still low (%.2f%%); buffering points by 50m in metric CRS for retry.",
                       rate * 100)
        buffered = points.to_crs(CRS.metric).copy()
        buffered["geometry"] = buffered.geometry.buffer(50)
        buffered = buffered.to_crs(lgas.crs)
        joined, rate = _attempt_join(buffered, "intersects", "buffered_intersects")

    if rate < min_success_rate:
        # prepare diagnostics
        rates_msg = "; ".join([f"{label}:{r:.2%}" for label, r, _ in attempts])
        pt_bounds = points.total_bounds
        lga_bounds = lgas.total_bounds
        best_join = max(attempts, key=lambda x: x[1])[2]
        unmatched = best_join[best_join["lga_name"].isna()].head(10)

        def _sample_lonlat(orig_points_gdf: gpd.GeoDataFrame, unmatched_index, n: int = 10):
            try:
                subset = orig_points_gdf.loc[unmatched_index]
            except Exception:
                return None
            subset = subset.head(n) if hasattr(subset, "head") else subset
            if subset.empty:
                return None
            pts = subset
            if pts.crs != CRS.wgs84:
                pts = pts.to_crs(CRS.wgs84)
            geom = pts.geometry
            if not (geom.geom_type == "Point").all():
                geom = geom.representative_point()
            return list(zip(geom.x.round(6), geom.y.round(6)))

        sample_points = _sample_lonlat(orig_points, unmatched.index, n=10)
        if sample_points is None:
            # fallback to centroids of unmatched join geometries
            fallback = unmatched
            if fallback.crs != CRS.wgs84:
                fallback = fallback.to_crs(CRS.wgs84)
            geom = fallback.geometry
            if not (geom.geom_type == "Point").all():
                geom = geom.centroid
            sample_points = list(zip(geom.x.round(6), geom.y.round(6)))

        raise ValueError(
            "Spatial join success rate too low after recovery attempts. "
            f"Rates tried: {rates_msg}. "
            f"Points bounds: {pt_bounds}. LGA bounds: {lga_bounds}. "
            f"Sample unmatched (lon, lat): {sample_points}"
        )

    LOGGER.info("Spatial join success rate: %.2f%%", rate * 100)
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

    facilities = _ensure_crs(facilities_gdf, CRS.wgs84).copy()
    lgas = _ensure_crs(lga_gdf, CRS.wgs84).copy()

    # Stable unique identifier for alignment
    if "lga_id" not in lgas.columns:
        lgas["lga_id"] = np.arange(len(lgas))

    base_cols = ["lga_id", "lga_name"]
    if "state_name" in lgas.columns:
        base_cols.append("state_name")
    base = lgas[base_cols].copy()

    # Facilities count per LGA (within, WGS84)
    facilities_joined = gpd.sjoin(facilities, lgas[["lga_id", "lga_name", "geometry"]], how="left", predicate="within")
    counts = facilities_joined.groupby("lga_id").size().rename("facilities_count").reset_index()

    # Distance proxy using centroids in metric CRS
    lgas_m = lgas.to_crs(CRS.metric)
    lga_centroids_m = lgas_m[["lga_id", "lga_name", "geometry"]].copy()
    if "state_name" in lgas_m.columns:
        lga_centroids_m["state_name"] = lgas_m["state_name"]
    lga_centroids_m["centroid_geom"] = lga_centroids_m.geometry.centroid

    facilities_m = facilities.to_crs(CRS.metric)
    facilities_m = facilities_m[
        facilities_m.geometry.notna()
        & ~facilities_m.geometry.is_empty
        & (facilities_m.geometry.geom_type == "Point")
    ].copy()

    if facilities_m.empty:
        lga_centroids_m["avg_distance_km_proxy"] = np.nan
    else:
        nearest = gpd.sjoin_nearest(
            lga_centroids_m.set_geometry("centroid_geom"),
            facilities_m,
            how="left",
            distance_col="dist_m",
        )
        dist_by_lga = nearest.groupby("lga_id")["dist_m"].min()
        lga_centroids_m["avg_distance_km_proxy"] = lga_centroids_m["lga_id"].map(dist_by_lga) / 1000.0

    metrics = base.merge(
        lga_centroids_m[["lga_id", "avg_distance_km_proxy"]],
        on="lga_id",
        how="left",
    ).merge(counts, on="lga_id", how="left")

    metrics["facilities_count"] = metrics["facilities_count"].fillna(0).astype(int)
    if population_df is not None and "population" in population_df.columns:
        pop = population_df[["lga_name", "population"]].copy()
        metrics = metrics.merge(pop, on="lga_name", how="left")
        metrics["facilities_per_10k"] = metrics["facilities_count"] / (metrics["population"] / 10000.0)
    else:
        metrics["facilities_per_10k"] = np.nan

    dist_non_null = metrics["avg_distance_km_proxy"].notna().mean() * 100 if len(metrics) else 0
    LOGGER.info(
        "Facility metrics: %d LGAs, %d facilities; distances available for %.1f%% of LGAs.",
        len(metrics),
        len(facilities_m),
        dist_non_null,
    )

    return metrics


def aggregate_tower_metrics_by_lga(
    towers_gdf: gpd.GeoDataFrame,
    lga_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate OpenCellID towers to LGAs and compute proximity metrics."""

    towers = _ensure_crs(towers_gdf, CRS.wgs84)
    lgas = _ensure_crs(lga_gdf, CRS.wgs84).copy()

    if "lga_uid" not in lgas.columns:
        raise ValueError("lga_uid column required on LGA GeoDataFrame for tower aggregation.")

    # Ensure area
    if "area_sq_km" not in lgas.columns:
        lgas_metric = lgas.to_crs(CRS.metric)
        lgas["area_sq_km"] = lgas_metric.geometry.area / 1_000_000.0

    # Counts
    join_counts = gpd.sjoin(
        towers[["geometry"]],
        lgas[["lga_uid", "geometry"]],
        how="left",
        predicate="within",
    )
    counts = join_counts.groupby("lga_uid").size().rename("towers_count")
    counts = counts.reindex(lgas["lga_uid"]).fillna(0).astype(int)

    # Distances to nearest tower
    lgas_centroids = lgas.to_crs(CRS.metric).copy()
    lgas_centroids["geometry"] = lgas_centroids.geometry.centroid
    towers_metric = towers.to_crs(CRS.metric)

    if towers_metric.empty:
        dist_series = pd.Series(np.nan, index=lgas["lga_uid"], name="avg_dist_to_tower_km")
    else:
        nearest = gpd.sjoin_nearest(
            lgas_centroids[["lga_uid", "geometry"]],
            towers_metric[["geometry"]],
            how="left",
            distance_col="dist_m",
        )
        dist_by_uid = nearest.groupby("lga_uid")["dist_m"].min() / 1000.0
        dist_series = lgas["lga_uid"].map(dist_by_uid)
        dist_series.name = "avg_dist_to_tower_km"

    metrics = pd.DataFrame({
        "lga_uid": lgas["lga_uid"],
        "towers_count": counts.values,
        "tower_density_per_km2": np.where(lgas["area_sq_km"] > 0, counts.values / lgas["area_sq_km"], np.nan),
        "avg_dist_to_tower_km": dist_series.values,
    })

    coverage_pct = (metrics["towers_count"] > 0).mean() * 100 if len(metrics) else 0.0
    LOGGER.info(
        "Tower metrics: %d LGAs, mean towers=%.2f, pct with >=1=%.1f%%",
        len(metrics),
        metrics["towers_count"].mean() if len(metrics) else 0,
        coverage_pct,
    )
    return metrics


def coverage_within_km(
    lga_gdf: gpd.GeoDataFrame,
    facilities_gdf: gpd.GeoDataFrame,
    km: float = 5.0,
    population_raster: Optional[str] = None,
) -> pd.DataFrame:
    """Estimate coverage within km. Uses area coverage if population raster is missing."""

    lgas = _ensure_crs(lga_gdf, CRS.wgs84).to_crs(CRS.metric)
    if "lga_id" not in lgas.columns:
        lgas["lga_id"] = np.arange(len(lgas))
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
                    coverage.append({"lga_id": row["lga_id"], "lga_name": row["lga_name"], "population_covered_pct": pct})
            return pd.DataFrame(coverage)

    for _, row in lgas.iterrows():
        lga_area = row.geometry.area
        covered_area = buffer_geom.intersection(row.geometry).area.sum()
        pct = (covered_area / lga_area) * 100 if lga_area > 0 else np.nan
        coverage.append({"lga_id": row["lga_id"], "lga_name": row["lga_name"], "area_covered_pct": pct})
    return pd.DataFrame(coverage)


def make_points_from_latlon(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> gpd.GeoDataFrame:
    """Create GeoDataFrame from lat/lon."""

    if lat_col not in df or lon_col not in df:
        raise ValueError(f"Missing lat/lon columns: {lat_col}, {lon_col}.")
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=CRS.wgs84)
