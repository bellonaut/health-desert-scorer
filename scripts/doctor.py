"""Environment and pipeline sanity checker for Stage A."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _check_import(pkg: str) -> str:
    try:
        __import__(pkg)
        return "ok"
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"missing ({exc.__class__.__name__})"


def _list_dir(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.iterdir())


def _tail(path: Path, n: int = 30) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


def main() -> int:
    print("== Environment ==")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for pkg in ("geopandas", "rasterio", "rasterstats"):
        print(f"Import {pkg}: {_check_import(pkg)}")

    print("\n== Data folders ==")
    print("data/raw:", _list_dir(Path("data/raw")))
    print("data/processed:", _list_dir(Path("data/processed")))

    print("\n== Build features ==")
    cmd = [sys.executable, "-m", "src.data.build_features"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "SUCCESS" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"Command: {' '.join(cmd)} -> {status}")
    if result.stdout:
        print("-- stdout --")
        print(result.stdout.strip())
    if result.stderr:
        print("-- stderr --")
        print(result.stderr.strip())

    log_path = Path("logs/build_features.log")
    tail = _tail(log_path)
    if tail:
        print("\nLast 30 log lines (build_features.log):")
        print("\n".join(tail))
    else:
        print("\nNo build_features.log found.")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
