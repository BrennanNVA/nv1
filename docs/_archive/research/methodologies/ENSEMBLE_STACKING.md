# Ensemble Stacking Methodology

## Overview

Ensemble methods combine multiple models to improve prediction accuracy. Research shows ensembles achieve **3-5% higher returns** in 2024-2025 studies.

## Stacking vs Blending

### Stacking

Stacking uses a meta-learner to combine base model predictions:
1. Train base models with out-of-fold (OOF) predictions
2. Train meta-learner on OOF predictions
3. Retrain base models on full dataset
4. Meta-learner combines final predictions

### Blending

Blending uses weighted average of base model predictions (simpler, no meta-learner).

## Implementation

### Stacking

```python
from nova.models.ensemble import EnsembleStacker, MetaLearnerType, create_xgboost_model

# Create base models
base_models = [
    create_xgboost_model(n_estimators=100, max_depth=6),
    create_xgboost_model(n_estimators=150, max_depth=8),
]

# Create stacker
stacker = EnsembleStacker(
    base_models=base_models,
    meta_learner_type=MetaLearnerType.RIDGE,
    n_folds=5
)

# Train
stacker.fit(X_train, y_train)

# Predict
predictions = stacker.predict(X_test)
```

### Blending

```python
from nova.models.ensemble import EnsembleBlender

# Create blender
blender = EnsembleBlender(
    base_models=base_models,
    weights=[0.6, 0.4]  # Optional weights
)

# Train
blender.fit(X_train, y_train)

# Predict
predictions = blender.predict(X_test)
```

## Out-of-Fold Predictions

Critical for preventing leakage:

1. Split data into K folds
2. For each fold:
   - Train base models on training folds
   - Predict on validation fold (OOF)
3. Meta-learner trains on OOF predictions
4. Retrain base models on full dataset

## Meta-Learner Types

- **Linear**: Simple linear regression
- **Ridge**: Ridge regression (L2 regularization)
- **XGBoost**: Gradient boosting meta-learner

## Multiple Base Models

Support for different algorithms:
- XGBoost
- LightGBM (if available)
- CatBoost (if available)

## Feature Importance

Get meta-learner feature importance:

```python
importances = stacker.get_feature_importance()
print(importances)
# {'XGBoost': 0.6, 'LightGBM': 0.4}
```

## Research Findings

- Ensembles show 3-5% higher returns vs single models
- Stacking generally outperforms blending
- Ridge meta-learner often works well
- 3-5 base models optimal (diminishing returns beyond)

## References

- "Ensemble Methods in Machine Learning" (Academic research)
- 2024-2025 studies on ensemble trading strategies
