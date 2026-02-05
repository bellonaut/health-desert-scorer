"""Export DHS cluster-level under-5 mortality labels for Nigeria."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


BR_COLUMNS = {
    "v001",
    "v005",
    "v008",
    "v024",
    "v025",
    "b3",
    "b5",
    "b7",
}

GPS_CLUSTER_CANDIDATES = ("DHSCLUST", "CLUSTER", "CLUST", "cluster", "dhsclust")
GPS_LAT_CANDIDATES = ("LATNUM", "LAT", "LATITUDE", "Y", "lat", "latitude")
GPS_LON_CANDIDATES = ("LONGNUM", "LON", "LONGITUDE", "X", "lon", "longitude")
GPS_URBAN_CANDIDATES = ("URBAN_RURA", "URBAN_RUR", "URBAN", "URBANRUR", "urban_rural")


def _lower_map(columns: list[str]) -> dict[str, str]:
    return {col.lower(): col for col in columns}


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = _lower_map(columns)
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in columns:
        for candidate in candidates:
            if candidate.lower() in column.lower():
                return column
    return None


def _ensure_columns(df: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required BR columns: {sorted(missing)}")
    return df


def _map_urban(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        norm = value.strip().upper()
        if norm in {"U", "URBAN"}:
            return "U"
        if norm in {"R", "RURAL"}:
            return "R"
    try:
        numeric = int(value)
    except (ValueError, TypeError):
        return None
    if numeric == 1:
        return "U"
    if numeric == 2:
        return "R"
    return None


def _map_region(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"north central", "north east", "north west"}:
            return "north"
        if norm in {"south east", "south south", "south west"}:
            return "south"
    try:
        numeric = int(value)
    except (ValueError, TypeError):
        return None
    mapping = {1: "north", 2: "north", 3: "north", 4: "south", 5: "south", 6: "south"}
    return mapping.get(numeric)


def _mode(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return None
    return series.value_counts().index[0]


def _load_br(br_path: Path) -> pd.DataFrame:
    df = pd.read_stata(br_path, convert_categoricals=True)
    df = _ensure_columns(df, BR_COLUMNS)
    df = df.dropna(subset=["v001"])
    df["age_months"] = df["v008"] - df["b3"]
    df = df[df["age_months"].between(0, 59)]
    df["weight"] = df["v005"] / 1_000_000
    return df


def _aggregate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("v001", as_index=False).agg(
        live_births=("weight", "sum"),
        u5_deaths=(
            "weight",
            lambda s: s[(df.loc[s.index, "b5"] == 0) & (df.loc[s.index, "b7"].notna()) & (df.loc[s.index, "b7"] < 60)].sum(),
        ),
        v025_mode=("v025", _mode),
    )
    grouped["u5mr"] = grouped["u5_deaths"] / grouped["live_births"] * 1000.0
    grouped["cluster_id"] = grouped["v001"].astype(int)
    grouped["quality_flag"] = np.where(grouped["live_births"] < 50, "sparse", "complete")
    grouped["urban_rural"] = grouped["v025_mode"].map(_map_urban)
    return grouped.drop(columns=["v001"])


def _load_gps(gps_path: Path) -> pd.DataFrame:
    gps = gpd.read_file(gps_path)
    cluster_col = _find_column(gps.columns.tolist(), GPS_CLUSTER_CANDIDATES)
    lat_col = _find_column(gps.columns.tolist(), GPS_LAT_CANDIDATES)
    lon_col = _find_column(gps.columns.tolist(), GPS_LON_CANDIDATES)
    if not cluster_col or not lat_col or not lon_col:
        raise ValueError(
            "GPS file missing required columns for cluster/lat/lon. "
            f"Found: {gps.columns.tolist()}"
        )
    urban_col = _find_column(gps.columns.tolist(), GPS_URBAN_CANDIDATES)
    gps = gps.rename(columns={cluster_col: "cluster_id", lat_col: "latitude", lon_col: "longitude"})
    gps = gps[["cluster_id", "latitude", "longitude"] + ([urban_col] if urban_col else [])]
    gps["cluster_id"] = gps["cluster_id"].astype(int)
    if urban_col:
        gps["urban_rural"] = gps[urban_col].map(_map_urban)
    return gps


def _validate_u5mr(national_u5mr: float, year: int) -> None:
    thresholds = {2013: (128, 148), 2018: (117, 132)}
    if year not in thresholds:
        return
    lower, upper = thresholds[year]
    if not (lower <= national_u5mr <= upper):
        raise ValueError(
            f"National U5MR {year} out of range ({lower}-{upper}): {national_u5mr:.1f}"
        )


def _validate_regional(df: pd.DataFrame) -> None:
    df = df.copy()
    df["region_group"] = df["v024"].map(_map_region)
    north = df[df["region_group"] == "north"]
    south = df[df["region_group"] == "south"]
    north_u5mr = north["weight"].sum()
    south_u5mr = south["weight"].sum()
    north_deaths = north[
        (north["b5"] == 0) & north["b7"].notna() & (north["b7"] < 60)
    ]["weight"].sum()
    south_deaths = south[
        (south["b5"] == 0) & south["b7"].notna() & (south["b7"] < 60)
    ]["weight"].sum()
    north_rate = north_deaths / north_u5mr * 1000.0 if north_u5mr else np.nan
    south_rate = south_deaths / south_u5mr * 1000.0 if south_u5mr else np.nan
    print(f"North U5MR: {north_rate:.1f} | South U5MR: {south_rate:.1f}")
    if np.isfinite(north_rate) and np.isfinite(south_rate):
        if north_rate < south_rate * 1.1:
            print(
                "WARNING: North U5MR is less than South U5MR +10%. "
                "Check region mapping and BR inputs."
            )


def export_clusters(br_path: Path, gps_path: Path, output_path: Path, year: int) -> pd.DataFrame:
    br = _load_br(br_path)

    total_births = br["weight"].sum()
    total_deaths = br[(br["b5"] == 0) & br["b7"].notna() & (br["b7"] < 60)]["weight"].sum()
    national_u5mr = total_deaths / total_births * 1000.0 if total_births else np.nan
    print(f"National U5MR {year}: {national_u5mr:.1f}")
    _validate_u5mr(national_u5mr, year)
    _validate_regional(br)

    clusters = _aggregate_clusters(br)

    gps = _load_gps(gps_path)
    merged = clusters.merge(gps, on="cluster_id", how="left")

    gps_coverage = merged["latitude"].notna().mean()
    print(f"GPS coverage: {gps_coverage:.1%}")
    if gps_coverage < 0.95:
        print("WARNING: GPS coverage below 95% of BR clusters.")
    if gps_coverage < 0.90:
        print("WARNING: GPS coverage below 90% of BR clusters.")

    if "urban_rural_y" in merged.columns and "urban_rural_x" in merged.columns:
        merged["urban_rural"] = merged["urban_rural_y"].combine_first(merged["urban_rural_x"])
    elif "urban_rural_x" in merged.columns:
        merged["urban_rural"] = merged["urban_rural_x"]
    elif "urban_rural_y" in merged.columns:
        merged["urban_rural"] = merged["urban_rural_y"]
    merged = merged.drop(columns=["urban_rural_x", "urban_rural_y"], errors="ignore")
    merged["quality_flag"] = np.where(
        merged["latitude"].isna() | merged["longitude"].isna(),
        "no_gps",
        merged["quality_flag"],
    )
    merged = merged[merged["quality_flag"] != "no_gps"].copy()
    merged["survey_year"] = year
    output = merged[
        [
            "cluster_id",
            "latitude",
            "longitude",
            "urban_rural",
            "live_births",
            "u5_deaths",
            "u5mr",
            "survey_year",
            "quality_flag",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} clusters to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DHS cluster-level U5MR labels.")
    parser.add_argument("--br", type=Path, required=True, help="Path to DHS BR .dta file")
    parser.add_argument("--gps", type=Path, required=True, help="Path to DHS GPS shapefile")
    parser.add_argument("--year", type=int, required=True, help="Survey year (2013 or 2018)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., data/interim/dhs_clusters_2018.csv)",
    )
    args = parser.parse_args()

    export_clusters(args.br, args.gps, args.output, args.year)


if __name__ == "__main__":
    main()
