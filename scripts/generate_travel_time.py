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
    python scripts/generate_travel_time.py --local

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
import rasterio
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
LOCAL_ORS_ISOCHRONE_URL = "http://localhost:8080/ors/v2/isochrones/{profile}"
INTERVALS_SEC = [30 * 60, 60 * 60]  # 30, 60 minutes (ORS free tier max is 3600s)
BATCH_SIZE = 5         # ORS free tier max locations per request
BATCH_SIZE_LOCAL = 5   # Local ORS batch size tuned to avoid request timeouts
RATE_LIMIT_DELAY = 3.0 # seconds between batches (20 req/min safe)
RATE_LIMIT_DELAY_LOCAL = 0.5
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
PROFILE = "driving-car"
YEAR = 2020


def _ors_api_key(cli_key: str | None, local: bool = False) -> str:
    if local:
        LOGGER.info("LOCAL MODE: using ORS at http://localhost:8080 without API key")
        return ""
    key = cli_key or os.getenv("ORS_API_KEY", "")
    if not key:
        raise SystemExit(
            "ORS API key required.\n"
            "Pass --api-key or set ORS_API_KEY env var.\n"
            "Get a free key at: https://openrouteservice.org/dev/#/signup"
        )
    return key


def deduplicate_facilities(gdf: gpd.GeoDataFrame, grid_km: float = 1.0) -> gpd.GeoDataFrame:
    """Snap facilities to a coarse grid and drop duplicates to avoid redundant ORS calls."""
    deg = grid_km / 111.0
    deduped = gdf.copy()
    deduped["_grid_lon"] = (deduped["_lon"] / deg).round() * deg
    deduped["_grid_lat"] = (deduped["_lat"] / deg).round() * deg
    deduped = deduped.drop_duplicates(subset=["_grid_lon", "_grid_lat"]).reset_index(drop=True)
    return deduped


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
        gdf = deduplicate_facilities(gdf, grid_km=1.0)
        LOGGER.info("After 1km deduplication: %d facilities", len(gdf))
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
        gdf = deduplicate_facilities(gdf, grid_km=1.0)
        LOGGER.info("After 1km deduplication: %d facilities", len(gdf))
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
        gdf = gpd.GeoDataFrame(
            demo,
            geometry=gpd.points_from_xy(demo["_lon"], demo["_lat"]),
            crs="EPSG:4326",
        )
        gdf = deduplicate_facilities(gdf, grid_km=1.0)
        LOGGER.info("After 1km deduplication: %d facilities", len(gdf))
        return gdf


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


def _checkpoint_exists(batch_idx: int) -> bool:
    return _checkpoint_path(batch_idx).exists()


def request_isochrones(
    locations: list[list[float]],
    api_key: str,
    profile: str = PROFILE,
    intervals_sec: list[int] = INTERVALS_SEC,
    retries: int = MAX_RETRIES,
    local: bool = False,
) -> list[dict] | None:
    """Request isochrones for up to 5 locations. Returns GeoJSON features or None on failure."""
    url = (LOCAL_ORS_ISOCHRONE_URL if local else ORS_ISOCHRONE_URL).format(profile=profile)
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = api_key
    body = {
        "locations": locations,
        "range": intervals_sec,
        "range_type": "time",
        "attributes": ["total_pop"],
        "smoothing": 0.25,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                try:
                    payload = json.loads(resp.content)
                except MemoryError:
                    size_mb = len(resp.content) / (1024 * 1024)
                    LOGGER.error(
                        "ORS response too large to decode in memory (%.1f MiB). Skipping batch.",
                        size_mb,
                    )
                    return None
                except json.JSONDecodeError as exc:
                    LOGGER.error("Invalid ORS JSON response: %s", exc)
                    return None
                return payload.get("features", [])
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


def _merge_features_into_unions(
    interval_stacks: dict[int, list[object | None]],
    features: list[dict],
) -> int:
    batch_groups: dict[int, list] = {iv: [] for iv in INTERVALS_SEC}
    merged = 0

    for feat in features:
        val = feat.get("properties", {}).get("value")
        if val is None:
            continue
        iv = int(val)
        if iv not in batch_groups:
            continue
        batch_groups[iv].append(shape(feat["geometry"]))
        merged += 1

    for iv, shapes in batch_groups.items():
        if not shapes:
            continue
        batch_union = unary_union(shapes) if len(shapes) > 1 else shapes[0]
        _push_union(interval_stacks[iv], batch_union)

    return merged


def _push_union(stack: list[object | None], geom: object) -> None:
    """Merge batch unions in a balanced stack to avoid repeated giant unary_union calls."""
    level = 0
    current = geom

    while True:
        if level >= len(stack):
            stack.append(current)
            return
        if stack[level] is None:
            stack[level] = current
            return
        current = unary_union([stack[level], current])
        stack[level] = None
        level += 1


def _finalize_union_stack(stack: list[object | None]) -> object | None:
    parts = [geom for geom in stack if geom is not None]
    if not parts:
        return None
    return unary_union(parts) if len(parts) > 1 else parts[0]


def _load_interval_unions_from_checkpoints(total_batches: int) -> tuple[dict[int, object | None], int]:
    interval_stacks: dict[int, list[object | None]] = {iv: [] for iv in INTERVALS_SEC}
    total_features = 0
    loaded_batches = 0

    LOGGER.info("Consolidating saved checkpoint coverage across %d batches", total_batches)

    for idx in range(total_batches):
        cached = _load_checkpoint(idx)
        if cached is None:
            continue
        total_features += _merge_features_into_unions(interval_stacks, cached)
        loaded_batches += 1
        if loaded_batches % 250 == 0:
            LOGGER.info(
                "Checkpoint merge: %d/%d saved batches consolidated",
                loaded_batches,
                total_batches,
            )

    interval_unions = {
        iv: _finalize_union_stack(stack)
        for iv, stack in interval_stacks.items()
    }
    return interval_unions, total_features


def build_facility_isochrones(
    facilities: gpd.GeoDataFrame,
    api_key: str,
    local: bool = False,
    max_batches: int | None = None,
    checkpoints_only: bool = False,
) -> tuple[dict[int, object | None], int]:
    """Request missing ORS batches, then stream saved checkpoints into bounded-memory unions."""
    coords = [[float(lon), float(lat)] for lon, lat in zip(facilities["_lon"], facilities["_lat"])]
    batch_size = BATCH_SIZE_LOCAL if local else BATCH_SIZE
    batches = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
        LOGGER.info("Calibration mode: capped at %d batches", max_batches)
    total_batches = len(batches)
    existing_count = sum(1 for idx in range(total_batches) if _checkpoint_exists(idx))
    missing = [(idx, batches[idx]) for idx in range(total_batches) if not _checkpoint_exists(idx)]

    LOGGER.info("Requesting isochrones for %d facilities in %d batches", len(coords), total_batches)
    LOGGER.info(
        "Resume state: %d/%d checkpoints present; %d missing batches remain",
        existing_count,
        total_batches,
        len(missing),
    )

    if checkpoints_only:
        LOGGER.info("Checkpoint-only mode: skipping ORS requests and consolidating saved checkpoints only")
        missing = []

    completed_attempts = 0
    for idx, batch in missing:
        features = request_isochrones(batch, api_key, local=local)
        completed_attempts += 1

        if features is None:
            LOGGER.warning("Batch %d failed after retries. Skipping.", idx)
        else:
            _save_checkpoint(idx, features)

        if completed_attempts % 50 == 0:
            LOGGER.info(
                "Progress: %d/%d batches complete",
                existing_count + completed_attempts,
                total_batches,
            )

        if idx < total_batches - 1:
            time.sleep(RATE_LIMIT_DELAY_LOCAL if local else RATE_LIMIT_DELAY)

    if missing and completed_attempts % 50 != 0:
        LOGGER.info(
            "Progress: %d/%d batches complete",
            existing_count + completed_attempts,
            total_batches,
        )

    interval_unions, total_features = _load_interval_unions_from_checkpoints(total_batches)
    LOGGER.info("Collected %d isochrone features total", total_features)
    return interval_unions, total_features


def compute_lga_coverage(
    lga_gdf: gpd.GeoDataFrame,
    interval_unions: dict[int, object | None],
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
        with rasterio.open(str(worldpop_path)) as src:
            raster_nodata = src.nodata if src.nodata is not None else 0
        LOGGER.info("WorldPop nodata value: %s", raster_nodata)

    # Pre-compute total population per LGA from raster
    lga_total_pop: dict[str, float] = {}
    if use_raster:
        LOGGER.info("Computing total population per LGA from WorldPop raster...")
        total_stats = zonal_stats(
            lga_gdf.__geo_interface__,
            str(worldpop_path),
            stats=["sum"],
            nodata=raster_nodata,
            all_touched=False,
        )
        for i, row in enumerate(lga_gdf.itertuples()):
            uid = str(row.lga_uid) if hasattr(row, "lga_uid") else str(i)
            lga_total_pop[uid] = float(total_stats[i].get("sum") or 0)

    for iv, union_geom in interval_unions.items():
        if union_geom is not None:
            LOGGER.info(
                "Interval %d min: merged coverage geometry ready",
                iv // 60,
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
                nodata=raster_nodata,
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


def main(
    api_key: str,
    dry_run: bool = False,
    local: bool = False,
    max_batches: int | None = None,
    checkpoints_only: bool = False,
) -> None:
    facilities = load_facilities()
    boundaries = load_boundaries()

    if dry_run:
        LOGGER.info("Dry run: using first 5 facilities only.")
        facilities = facilities.head(5)

    interval_unions, feature_count = build_facility_isochrones(
        facilities,
        api_key,
        local=local,
        max_batches=max_batches,
        checkpoints_only=checkpoints_only,
    )

    if feature_count == 0:
        LOGGER.error("No isochrone features returned. Check API key and facility coordinates.")
        return

    coverage_df = compute_lga_coverage(boundaries, interval_unions, WORLDPOP_PATH)

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
    parser.add_argument("--local", action="store_true", help="Use a local ORS instance at http://localhost:8080")
    parser.add_argument("--dry-run", action="store_true", help="Run with 5 facilities only to test pipeline")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many batches (for calibration)")
    parser.add_argument("--checkpoints-only", action="store_true", help="Skip ORS requests and rebuild coverage from saved checkpoints")
    args = parser.parse_args()
    main(
        _ors_api_key(args.api_key, local=args.local),
        dry_run=args.dry_run,
        local=args.local,
        max_batches=args.max_batches,
        checkpoints_only=args.checkpoints_only,
    )
