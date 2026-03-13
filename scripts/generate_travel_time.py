"""Generate routed travel-time coverage statistics per Nigerian LGA.

Method (replicating HeiGIT's open healthcare access pipeline):
  1. Load NHFR facility coordinates from data/raw/
  2. Request ORS isochrones at 30, 60, 120-minute driving intervals
     -- uses NHFR facilities rather than OSM to avoid Kano/Bauchi import issues
  3. Union isochrones per LGA boundary (geoBoundaries admin2)
  4. Intersect with WorldPop 2020 raster via rasterstats.zonal_stats
  5. Output: data/raw/travel_time_lga.csv
     columns: lga_id, lga_name, state_name, year, profile,
              pop_pct_30min, pop_pct_60min, pop_pct_120min,
              pop_covered_60min, source, generated_at

Usage:
    python scripts/generate_travel_time.py --api-key YOUR_ORS_KEY

ORS free tier: 500 isochrone requests/day, 20/minute.
  ~4,000 NHFR facilities = ~800 batches of 5 = 1.6 days at free tier.
  Checkpoint files allow resuming across sessions automatically.

Attribution:
    Methodology adapted from HeiGIT / openrouteservice healthcare access
    notebook (GIScience/openrouteservice-examples). Thank you to Lisa
    Shkredova and Marcel at HeiGIT for methodology guidance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape
from shapely.ops import unary_union

try:
    from rasterstats import zonal_stats
except ImportError:
    raise SystemExit(
        "rasterstats is required: pip install rasterstats\n"
        "Also ensure rasterio is installed: pip install rasterio"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "_ors_checkpoints"

OUTPUT_PATH = RAW_DIR / "travel_time_lga.csv"
NHFR_PATH = RAW_DIR / "nhfr_facilities.csv"
NHFR_GEOJSON_PATH = RAW_DIR / "health_facilities.geojson"
BOUNDARIES_PATH = RAW_DIR / "lga_boundaries.geojson"

# WorldPop raster — prefer 2024, fall back to earlier years if absent.
# 100m and 1km aggregated files are both valid; 1km is faster for LGA-level stats.
_WORLDPOP_CANDIDATES = [
    RAW_DIR / "nga_ppp_2024_100m_Aggregated.tif",
    RAW_DIR / "nga_ppp_2024_1km_Aggregated.tif",
    RAW_DIR / "nga_ppp_2020_1km_Aggregated.tif",
    RAW_DIR / "nga_ppp_2020_100m_Aggregated.tif",
]
WORLDPOP_PATH = next((p for p in _WORLDPOP_CANDIDATES if p.exists()), _WORLDPOP_CANDIDATES[0])
WORLDPOP_YEAR = int(WORLDPOP_PATH.name.split("_")[2]) if WORLDPOP_PATH.exists() else 2024

ORS_ISOCHRONE_URL = "https://api.openrouteservice.org/v2/isochrones/{profile}"
INTERVALS_SEC = [30 * 60, 60 * 60]  # 30, 60 minutes (ORS free tier max is 3600s)
BATCH_SIZE = 5         # ORS free tier max locations per request
RATE_LIMIT_DELAY = 3.0 # seconds between batches (20 req/min safe)
MAX_RETRIES = 3
PROFILE = "driving-car"
YEAR = 2020


def _ors_api_key(cli_key: str | None) -> str:
    key = cli_key or os.getenv("ORS_API_KEY", "")
    if not key:
        raise SystemExit(
            "ORS API key required.\n"
            "Pass --api-key or set ORS_API_KEY env var.\n"
            "Get a free key at: https://openrouteservice.org/dev/#/signup"
        )
    return key


def load_facilities() -> gpd.GeoDataFrame:
    """Load NHFR facility coordinates.

    Expects columns: longitude, latitude (or lon/lat).
    Falls back to a minimal demo set if NHFR file is absent.
    """
    if NHFR_PATH.exists():
        df = pd.read_csv(NHFR_PATH)
        lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long", "x")), None)
        lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat", "y")), None)
        if lon_col is None or lat_col is None:
            raise SystemExit(f"Cannot find lon/lat columns in {NHFR_PATH}. Found: {list(df.columns)}")
        df = df.dropna(subset=[lon_col, lat_col])
        df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
        df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df = df.dropna(subset=["_lon", "_lat"])
        # Nigeria bounding box guard
        df = df[
            df["_lon"].between(2.6, 14.7) &
            df["_lat"].between(4.0, 14.0)
        ]
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["_lon"], df["_lat"]),
            crs="EPSG:4326",
        )
        LOGGER.info("Loaded %d facilities from NHFR CSV", len(gdf))
        return gdf
    elif NHFR_GEOJSON_PATH.exists():
        gdf = gpd.read_file(NHFR_GEOJSON_PATH).to_crs("EPSG:4326")
        # Extract lon/lat from geometry
        gdf["_lon"] = gdf.geometry.x
        gdf["_lat"] = gdf.geometry.y
        # Nigeria bounding box guard
        gdf = gdf[
            gdf["_lon"].between(2.6, 14.7) &
            gdf["_lat"].between(4.0, 14.0)
        ]
        LOGGER.info("Loaded %d facilities from health_facilities.geojson", len(gdf))
        return gdf
    else:
        LOGGER.warning(
            "%s not found. Using 10-facility demo subset. "
            "Full run requires the NHFR CSV.",
            NHFR_PATH,
        )
        demo = pd.DataFrame({
            "_lon": [3.35, 7.49, 8.52, 3.90, 5.62, 11.85, 6.45, 9.08, 4.82, 13.31],
            "_lat": [6.60, 9.06, 11.97, 7.39, 5.86, 13.15, 10.52, 7.74, 8.21, 12.44],
        })
        return gpd.GeoDataFrame(
            demo,
            geometry=gpd.points_from_xy(demo["_lon"], demo["_lat"]),
            crs="EPSG:4326",
        )


def load_boundaries() -> gpd.GeoDataFrame:
    if not BOUNDARIES_PATH.exists():
        raise SystemExit(f"LGA boundaries not found at {BOUNDARIES_PATH}. Run download_open_data.py first.")
    gdf = gpd.read_file(BOUNDARIES_PATH)
    gdf = gdf.rename(columns={"lganame": "lga_name", "statename": "state_name", "uniq_id": "lga_uid"})
    for col in ("lga_name", "state_name"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).str.strip()
    gdf = gdf.to_crs("EPSG:4326")
    LOGGER.info("Loaded %d LGA boundaries", len(gdf))
    return gdf


def _checkpoint_path(batch_idx: int) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"batch_{batch_idx:05d}.json"


def _load_checkpoint(batch_idx: int) -> dict | None:
    p = _checkpoint_path(batch_idx)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_checkpoint(batch_idx: int, data: dict) -> None:
    _checkpoint_path(batch_idx).write_text(json.dumps(data))


def request_isochrones(
    locations: list[list[float]],
    api_key: str,
    profile: str = PROFILE,
    intervals_sec: list[int] = INTERVALS_SEC,
    retries: int = MAX_RETRIES,
) -> list[dict] | None:
    """Request isochrones for up to 5 locations. Returns GeoJSON features or None on failure."""
    url = ORS_ISOCHRONE_URL.format(profile=profile)
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "locations": locations,
        "range": intervals_sec,
        "range_type": "time",
        "attributes": ["total_pop"],
        "smoothing": 0.25,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("features", [])
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                LOGGER.warning("Rate limited. Waiting %ds before retry.", wait)
                time.sleep(wait)
                continue
            LOGGER.error("ORS error %d: %s", resp.status_code, resp.text[:200])
            return None
        except requests.RequestException as exc:
            LOGGER.warning("Request error (attempt %d/%d): %s", attempt + 1, retries, exc)
            time.sleep(5 * (attempt + 1))
    return None


def build_facility_isochrones(facilities: gpd.GeoDataFrame, api_key: str) -> list[dict]:
    """Batch all facilities through ORS. Returns flat list of GeoJSON features with interval metadata."""
    coords = [[float(lon), float(lat)] for lon, lat in zip(facilities["_lon"], facilities["_lat"])]
    batches = [coords[i:i + BATCH_SIZE] for i in range(0, len(coords), BATCH_SIZE)]
    all_features: list[dict] = []

    LOGGER.info("Requesting isochrones for %d facilities in %d batches", len(coords), len(batches))

    for idx, batch in enumerate(batches):
        cached = _load_checkpoint(idx)
        if cached is not None:
            all_features.extend(cached)
            continue

        features = request_isochrones(batch, api_key)
        if features is None:
            LOGGER.warning("Batch %d failed after retries. Skipping.", idx)
            continue

        _save_checkpoint(idx, features)
        all_features.extend(features)

        if idx < len(batches) - 1:
            time.sleep(RATE_LIMIT_DELAY)

        if (idx + 1) % 50 == 0:
            LOGGER.info("Progress: %d/%d batches complete", idx + 1, len(batches))

    LOGGER.info("Collected %d isochrone features total", len(all_features))
    return all_features


def isochrones_by_interval(features: list[dict]) -> dict[int, list]:
    """Group ORS features by their range value (seconds)."""
    groups: dict[int, list] = {iv: [] for iv in INTERVALS_SEC}
    for feat in features:
        val = feat.get("properties", {}).get("value")
        if val is None:
            continue
        iv = int(val)
        if iv in groups:
            groups[iv].append(shape(feat["geometry"]))
    return groups


def compute_lga_coverage(
    lga_gdf: gpd.GeoDataFrame,
    isochrone_groups: dict[int, list],
    worldpop_path: Path,
) -> pd.DataFrame:
    """For each LGA, compute population % within each travel-time interval.

    Method:
    - Union all facility isochrones at each interval into a single polygon
    - Intersect that polygon with each LGA boundary
    - Use rasterstats to sum WorldPop population inside each intersected zone
    - Divide by total LGA population from same raster
    """
    if not worldpop_path.exists():
        LOGGER.warning(
            "WorldPop raster not found at %s. "
            "Pop_pct values will be null. "
            "Download 2024 file: https://hub.worldpop.org/geodata/listing?id=49654",
            worldpop_path,
        )
        use_raster = False
    else:
        use_raster = True
        LOGGER.info("Using WorldPop raster: %s", worldpop_path.name)

    # Pre-compute total population per LGA from raster
    lga_total_pop: dict[str, float] = {}
    if use_raster:
        LOGGER.info("Computing total population per LGA from WorldPop raster...")
        total_stats = zonal_stats(
            lga_gdf.__geo_interface__,
            str(worldpop_path),
            stats=["sum"],
            nodata=0,
            all_touched=False,
        )
        for i, row in enumerate(lga_gdf.itertuples()):
            uid = str(row.lga_uid) if hasattr(row, "lga_uid") else str(i)
            lga_total_pop[uid] = float(total_stats[i].get("sum") or 0)

    # Union isochrones per interval
    interval_unions: dict[int, object] = {}
    for iv, shapes in isochrone_groups.items():
        if shapes:
            interval_unions[iv] = unary_union(shapes)
            LOGGER.info(
                "Interval %d min: unioned %d polygons",
                iv // 60,
                len(shapes),
            )

    records = []
    for row in lga_gdf.itertuples():
        uid = str(row.lga_uid) if hasattr(row, "lga_uid") else str(getattr(row, "Index", ""))
        lga_geom = row.geometry
        if lga_geom is None or lga_geom.is_empty:
            continue

        total_pop = lga_total_pop.get(uid, 0)
        rec: dict = {
            "lga_id": uid,
            "lga_name": getattr(row, "lga_name", ""),
            "state_name": getattr(row, "state_name", ""),
            "year": YEAR,
            "profile": PROFILE,
        }

        for iv in INTERVALS_SEC:
            col_name = f"pop_pct_{iv // 60}min"
            pop_col = f"pop_covered_{iv // 60}min"
            union_geom = interval_unions.get(iv)
            if union_geom is None:
                rec[col_name] = None
                rec[pop_col] = None
                continue

            # Intersect LGA with union of all isochrones at this interval
            try:
                intersection = lga_geom.intersection(union_geom)
            except Exception:
                rec[col_name] = None
                rec[pop_col] = None
                continue

            if intersection.is_empty or not use_raster:
                rec[col_name] = None
                rec[pop_col] = None
                continue

            inter_stats = zonal_stats(
                [intersection.__geo_interface__],
                str(worldpop_path),
                stats=["sum"],
                nodata=0,
                all_touched=False,
            )
            covered_pop = float(inter_stats[0].get("sum") or 0)
            rec[pop_col] = covered_pop
            rec[col_name] = (
                round(covered_pop / total_pop * 100, 2)
                if total_pop > 0
                else None
            )

        records.append(rec)

    df = pd.DataFrame(records)
    # Add 60-min alias used by data_api + build_features
    if "pop_pct_60min" in df.columns:
        df["pop_pct_within_60min"] = df["pop_pct_60min"]

    df["source"] = "ORS_isochrones_NHFR"
    df["worldpop_year"] = WORLDPOP_YEAR
    df["generated_at"] = datetime.now(timezone.utc).isoformat()
    return df


def validate_output(df: pd.DataFrame, dry_run: bool = False) -> None:
    issues = []
    # 120min was dropped — only validate intervals we actually request
    for col in ("pop_pct_30min", "pop_pct_60min"):
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue
        vals = df[col].dropna()
        if (vals < 0).any() or (vals > 100).any():
            issues.append(f"{col} has values outside [0, 100]")
    null_pct = df["pop_pct_60min"].isna().mean() if "pop_pct_60min" in df.columns else 1.0
    # Dry run uses 5 facilities so near-100% null is expected — skip threshold check
    if not dry_run and null_pct > 0.3:
        issues.append(
            f"pop_pct_60min is null for {null_pct:.0%} of LGAs — "
            "check WorldPop raster path or ORS coverage"
        )
    covered = int((1 - null_pct) * len(df))
    if issues:
        for issue in issues:
            LOGGER.warning("VALIDATION: %s", issue)
    else:
        LOGGER.info("Validation passed. %d / %d LGAs with pop_pct_60min.", covered, len(df))


def main(api_key: str, dry_run: bool = False) -> None:
    facilities = load_facilities()
    boundaries = load_boundaries()

    if dry_run:
        LOGGER.info("Dry run: using first 5 facilities only.")
        facilities = facilities.head(5)

    features = build_facility_isochrones(facilities, api_key)

    if not features:
        LOGGER.error("No isochrone features returned. Check API key and facility coordinates.")
        return

    isochrone_groups = isochrones_by_interval(features)
    coverage_df = compute_lga_coverage(boundaries, isochrone_groups, WORLDPOP_PATH)

    validate_output(coverage_df, dry_run=dry_run)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(OUTPUT_PATH, index=False)
    LOGGER.info("Saved travel-time coverage to %s (%d rows)", OUTPUT_PATH, len(coverage_df))

    if "pop_pct_60min" in coverage_df.columns:
        valid = coverage_df["pop_pct_60min"].dropna()
        if len(valid):
            LOGGER.info(
                "pop_pct_60min summary: %d LGAs covered | median=%.1f%% mean=%.1f%% min=%.1f%% max=%.1f%%",
                len(valid), valid.median(), valid.mean(), valid.min(), valid.max(),
            )
            if dry_run:
                covered_lgas = coverage_df[coverage_df["pop_pct_60min"].notna()][["lga_name", "state_name", "pop_pct_30min", "pop_pct_60min"]]
                LOGGER.info("Dry run covered LGAs:\n%s", covered_lgas.to_string(index=False))
        else:
            LOGGER.info("Dry run complete. No LGA coverage from 5 facilities — expected. Full run uses all %d facilities.", len(load_facilities()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ORS travel-time coverage per LGA")
    parser.add_argument("--api-key", default=None, help="ORS API key (or set ORS_API_KEY env var)")
    parser.add_argument("--dry-run", action="store_true", help="Run with 5 facilities only to test pipeline")
    args = parser.parse_args()
    main(_ors_api_key(args.api_key), dry_run=args.dry_run)