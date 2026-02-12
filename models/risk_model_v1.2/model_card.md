# Risk Model v1.2

## Intended Use
Planning tool for identifying LGAs with healthcare access barriers in Nigeria.

## NOT Intended For
- Clinical diagnosis
- Individual health risk prediction
- Sole basis for funding decisions without field validation

## Training Data
- **DHS Survey**: 2013, 2018
- **Facilities**: NHFR 2020
- **Population**: WorldPop 2020
- **Connectivity**: OpenCellID 2019

## Model Architecture
- **Algorithm**: Gradient boosted tree model (versioned artifact)
- **Features**: 6 core LGA-level features
- **Cross-Validation**: 5-fold stratified by state

## Performance Metrics
- **Accuracy**: 82%
- **Precision**: 0.79
- **Recall**: 0.84
- **F1 Score**: 0.81
- **ROC-AUC**: 0.88

## Known Limitations
- Does not account for security/conflict zones
- Seasonal road accessibility not captured
- Staff quality/availability not measured
- Survey sampling variation across regions

## Bias & Fairness
- Urban LGAs can be over-represented in registry data
- Data completeness varies by state
- Field validation is required before high-stakes decisions

## Update Schedule
- **Last Trained**: January 2026
- **Next Scheduled Update**: July 2026
