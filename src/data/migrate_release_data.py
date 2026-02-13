"""Migrate release-safe artifacts into data/bronze and write a SHA256 manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"

MAPPINGS = [
    (ROOT / "data" / "raw" / "lga_boundaries.geojson", BRONZE / "boundaries" / "lga_boundaries.geojson"),
    (ROOT / "data" / "raw" / "health_facilities.geojson", BRONZE / "facilities" / "health_facilities.geojson"),
    (ROOT / "data" / "raw" / "population_lga.csv", BRONZE / "population" / "population_lga.csv"),
    (ROOT / "data" / "processed" / "population_lga_canonical.csv", BRONZE / "population" / "population_lga_canonical.csv"),
    (ROOT / "data" / "processed" / "lga_features.csv", BRONZE / "derived" / "lga_features.csv"),
    (ROOT / "data" / "processed" / "lga_predictions.csv", BRONZE / "derived" / "lga_predictions.csv"),
    (ROOT / "data" / "processed" / "shap_values.csv", BRONZE / "derived" / "shap_values.csv"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    BRONZE.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, dest in MAPPINGS:
        if not source.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(
            {
                "source": str(source.relative_to(ROOT)),
                "path": str(dest.relative_to(ROOT)),
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
                "copied_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": copied,
    }
    (BRONZE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: migrated {len(copied)} artifacts into {BRONZE}")


if __name__ == "__main__":
    main()
