# Risk Model v1.3

## Intended Use
Planning tool for identifying LGAs with healthcare access barriers in Nigeria.

## Training Data
- DHS Survey: 2013, 2018, 2024
- Facilities: NHFR 2020
- Population: WorldPop-derived LGA totals
- Connectivity: OpenCellID

## Validation
- Train split: 2013 + 2018
- Temporal holdout: 2024

## Model Architecture
- Algorithm: Gradient boosted tree classifier wrapped to emit 0-10 risk scores
- Features: 6 core LGA-level inputs

## Known Limitations
- Does not capture seasonal road access shocks
- Registry completeness still varies by state
- Scores remain planning aids and require local validation
