"""Create simplified boundary files for performance."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
RAW = ROOT / "data" / "raw"


def _load_source() -> gpd.GeoDataFrame:
    silver_master = SILVER / "lga_master.geojson"
    if silver_master.exists():
        return gpd.read_file(silver_master)
    raw_boundaries = RAW / "lga_boundaries.geojson"
    if raw_boundaries.exists():
        return gpd.read_file(raw_boundaries)
    raise FileNotFoundError("Missing lga boundaries for simplification.")


def _report(path: Path) -> None:
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:,.1f} KB)")


def main() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)
    lgas = _load_source()

    hi = lgas.copy()
    hi["geometry"] = hi["geometry"].simplify(0.005, preserve_topology=True)
    hi_path = SILVER / "lga_boundaries_hi.geojson"
    hi.to_file(hi_path, driver="GeoJSON")
    _report(hi_path)

    med = lgas.copy()
    med["geometry"] = med["geometry"].simplify(0.01, preserve_topology=True)
    med_path = SILVER / "lga_boundaries_med.geojson"
    med.to_file(med_path, driver="GeoJSON")
    _report(med_path)

    low = lgas.copy()
    low["geometry"] = low["geometry"].simplify(0.02, preserve_topology=True)
    low_path = SILVER / "lga_boundaries_lo.geojson"
    low.to_file(low_path, driver="GeoJSON")
    _report(low_path)

    if "state_id" in lgas.columns:
        states = lgas.dissolve(by="state_id", aggfunc="first")
    elif "statename" in lgas.columns:
        states = lgas.dissolve(by="statename", aggfunc="first")
    else:
        states = lgas.copy()
    states["geometry"] = states["geometry"].simplify(0.03, preserve_topology=True)
    state_path = SILVER / "state_boundaries.geojson"
    states.to_file(state_path, driver="GeoJSON")
    _report(state_path)


if __name__ == "__main__":
    main()
