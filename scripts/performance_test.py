"""Simple performance budget check for core operations."""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_api import load_backend_data, filter_geo


def measure_load_times() -> dict[str, float]:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    geo_df, _ = load_backend_data(boundary_resolution="low", is_mobile=True)
    timings["data_load"] = time.perf_counter() - start
    timings["geometry_load_low"] = timings["data_load"]

    start = time.perf_counter()
    load_backend_data(boundary_resolution="high", is_mobile=False)
    timings["geometry_load_high"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = filter_geo(geo_df, state_filter="Kano", year=2018)
    timings["filter_state"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = geo_df.sort_values("risk_score", ascending=False)
    timings["ranking"] = time.perf_counter() - start

    return timings


def check_performance_budget(timings: dict[str, float]) -> list[str]:
    budget = {
        "data_load": 1.0,
        "geometry_load_low": 0.5,
        "geometry_load_high": 2.0,
        "filter_state": 0.1,
        "ranking": 0.1,
    }

    violations = []
    for operation, limit in budget.items():
        actual = timings.get(operation, 0)
        if actual > limit:
            violations.append(f"{operation}: {actual:.3f}s (limit: {limit}s)")
    return violations


def main() -> int:
    print("Running performance tests...")
    timings = measure_load_times()

    Path("logs").mkdir(parents=True, exist_ok=True)
    (Path("logs") / "perf_results.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    for op, time_taken in timings.items():
        print(f"{op}: {time_taken:.3f}s")

    violations = check_performance_budget(timings)
    if violations:
        print("\nPerformance budget violations:")
        for v in violations:
            print(f"- {v}")
        return 1

    print("OK: All operations within performance budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
