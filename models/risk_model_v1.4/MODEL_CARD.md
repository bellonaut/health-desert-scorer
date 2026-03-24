# HDS Nigeria Risk Model v1.4

## Summary
Version 1.4 keeps the all-year base scorer on the shared access and mortality feature set, then adds a 2024-only routed travel-time recalibration so LGAs with weak road access move upward without inventing historical drive-time values for 2013 or 2018.

## Training data
- DHS 2013, 2018 (train)
- DHS 2024 (temporal holdout)

## Features
### Stage 1 (all years)
- facilities_per_10k
- avg_distance_km
- u5mr_mean
- coverage_5km
- towers_per_10k
- population_density

### Stage 2 - 2024 only
- pop_pct_60min: % of LGA population within 60-min drive of any health facility
  Source: OpenRouteService isochrones against NHFR 51,022 facilities (1km deduplicated to 35,732)
  Coverage: 2024 only. Not available for 2013/2018.

## Evaluation (2024 holdout)
| Metric | v1.3 | v1.4 |
|--------|------|------|
| Spearman rho | 0.612 | 0.941 |
| MAE | 4.217 | 0.435 |
| RMSE | 4.408 | 0.545 |
| R2 | -7.703 | 0.867 |

## 2024 SHAP top 5
- u5mr_mean: 0.8474
- facilities_per_10k: 0.6116
- avg_distance_km: 0.4182
- pop_pct_60min: 0.3160
- towers_per_10k: 0.2234

## Known limitations
- pop_pct_60min uses driving-car profile only. Walking/motorcycle access not modelled. May understate access barriers in areas where vehicles are uncommon.
- OSM road network quality varies by region. Northwest Nigeria road data is less complete than Southwest.
- 2013 and 2018 scores do not incorporate travel time. Cross-year comparisons should account for this methodological difference.

## Deployment
- Scores are rank-normalized within year to [0, 10]
- Risk score > 5.5 = above national median (higher risk)
- Risk score <= 5.5 = below national median (lower risk)
