# Drift Detection Methodology

## Overview

Drift detection monitors model performance decay and data distribution shifts. This document describes the comprehensive drift detection system implemented in Nova Aetus.

## Types of Drift

### 1. Data Drift

Distribution shift in input features:
- **PSI (Population Stability Index)**: Measures distribution shift
- **KS Test**: Kolmogorov-Smirnov test for distribution differences

### 2. Concept Drift

Change in relationship between features and target:
- Detected via reverse CV and performance decay
- IC (Information Coefficient) drop over time

### 3. Prediction Drift

Distribution shift in model predictions:
- KS test on prediction distributions
- Mean shift detection

### 4. Target Drift

Distribution shift in target variable (if available).

## Implementation

### Data Drift Detection

```python
from nova.monitoring import DriftDetector
import polars as pl

# Initialize detector with reference data
detector = DriftDetector(
    reference_data=training_data,
    psi_threshold=0.2,
    ks_threshold=0.05
)

# Detect drift
result = detector.detect_data_drift(current_data)

if result.drift_detected:
    print(f"Drift detected: {result.severity}")
    print(f"Drifted features: {result.drifted_features}")
    print(f"PSI: {result.psi:.4f}")
```

### Concept Drift Detection

```python
# Record predictions and actuals
detector.record_prediction_actual(
    prediction=0.05,
    actual=0.03
)

# Detect concept drift
result = detector.detect_concept_drift(window_size=100)

if result.drift_detected:
    print(f"Concept drift: {result.severity}")
    print(f"IC drop: {result.details['ic_drop']:.4f}")
```

### Prediction Drift Detection

```python
result = detector.detect_prediction_drift(
    current_predictions=current_preds,
    reference_predictions=reference_preds
)
```

### Comprehensive Check

```python
# Run all drift checks
results = detector.check_all_drift(
    current_data=current_data,
    current_predictions=current_preds
)

for drift_type, result in results.items():
    if result.drift_detected:
        print(f"{drift_type} drift detected: {result.severity}")
```

## PSI Calculation

Population Stability Index:

```
PSI = sum((current_prob - reference_prob) * log(current_prob / reference_prob))
```

Thresholds:
- **PSI < 0.1**: No significant drift
- **PSI 0.1-0.2**: Minor drift
- **PSI > 0.2**: Significant drift

## Automated Alerts

Drift alerts are sent via Discord/notification service:

```python
from nova.core.notifications import NotificationService

detector.set_notification_service(notification_service)
```

## Integration with Prometheus

Metrics exported:
- `drift_psi_max`: Maximum PSI value
- `drift_ks_pvalue_min`: Minimum KS p-value
- `drift_features_count`: Number of drifted features
- `drift_detected_total`: Total drift detections

## Retraining Triggers

Drift detection can trigger automated retraining:

```python
if result.drift_detected and result.severity == "high":
    # Trigger retraining
    trigger_model_retraining()
```

## Evidently AI Integration

Optional integration with Evidently AI for comprehensive drift reports:

```python
# Evidently AI automatically used if available
# Provides detailed drift reports and visualizations
```

## References

- "Concept Drift Detection in Machine Learning" (Academic research)
- Evidently AI Documentation
- NannyML Documentation
