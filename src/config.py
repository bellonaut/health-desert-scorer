"""Project-wide configuration constants."""

from __future__ import annotations

# Feature validation thresholds
MIN_LGA_COUNT = 500
AVG_DISTANCE_KM_MAX = 200
POPULATION_MERGE_COVERAGE_MIN = 0.98

# Spatial join recovery
MIN_SPATIAL_JOIN_SUCCESS_RATE = 0.7
POINT_BUFFER_METERS = 50

# Coverage defaults
COVERAGE_KM_DEFAULT = 5.0
