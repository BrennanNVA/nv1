# Integration Status - Research Tools Enhancement

## Overview

This document outlines what's automatically integrated vs what requires manual setup for the research tools and documentation enhancements.

## ✅ Fully Integrated (No Action Needed)

1. **ExecutionCostModel**: Automatically created in `ExecutionEngine.__init__()` with defaults. Works out-of-the-box for cost estimation.

2. **SHAP in Trainer**: `ModelTrainer.save_model()` now includes SHAP calculation by default. Just pass `X_val` and `feature_names` when saving models.

3. **Enhanced Validation**: `WalkForwardOptimizer` now supports `gap_days` and `horizons` parameters - can be used by setting these when creating the optimizer.

4. **Dashboard Monitoring Tab**: New "Model Monitoring" tab automatically available in dashboard.

5. **Dependencies**: New libraries added to `requirements.txt` - need to install (see below).

## ⚠️ Partially Integrated (Optional Enhancement)

1. **IC Tracker**: Module exists but not instantiated in main loop. To use:
   - Instantiate `ICTracker` in `main.py`
   - Call `record_prediction()` and `record_actual()` in trading loop
   - Results available in Prometheus metrics

2. **Drift Detector**: Module exists but not instantiated. To use:
   - Instantiate `DriftDetector` with reference data
   - Call `check_all_drift()` periodically
   - Set notification service for alerts

3. **Correlation Regime Detection**: Available in `ConfluenceLayer` but requires passing `returns_df` to `detect_regime()`. Currently using basic regime detection.

4. **Ensemble Models**: Separate classes - need to explicitly use `EnsembleStacker` or `EnsembleBlender` instead of single `ModelTrainer`.

## 📋 Manual Steps Required

### 1. Install New Dependencies

```bash
cd nova_aetus
pip install -r requirements.txt
# Or specifically:
pip install mlfinlab>=1.0.0 shap>=0.45.0 evidently>=0.4.0 nannyml>=0.10.0 feast>=0.38.0
```

**Note**: `mlfinlab` may have dependencies that require specific setup. Check MLFinLab documentation.

### 2. Enable SHAP When Training Models

When training models, pass validation data to save_model:

```python
trainer.save_model(
    filepath="model.json",
    include_shap=True,  # Default True
    X_val=X_val_numpy,  # Required for SHAP
    feature_names=feature_names  # Optional but recommended
)
```

### 3. (Optional) Enable IC Tracking in Trading Loop

Add to `main.py` trading_loop function:

```python
from ..monitoring import ICTracker

# In main(), after creating services:
ic_tracker = ICTracker(horizons=[1, 5, 20])

# In trading loop, after prediction:
ic_tracker.record_prediction("technical_signal", horizon=1, prediction=prediction)

# Later, when actual return is known:
ic_tracker.record_actual("technical_signal", horizon=1, actual=actual_return)
```

### 4. (Optional) Enable Drift Detection

Add to `main.py`:

```python
from ..monitoring import DriftDetector

# After loading training data as reference:
drift_detector = DriftDetector(reference_data=training_features_df)
drift_detector.set_notification_service(notifications)

# Periodically check (e.g., daily):
drift_results = drift_detector.check_all_drift(
    current_data=current_features_df,
    current_predictions=current_predictions
)
```

### 5. (Optional) Use Ensemble Models

Replace single model training with ensemble:

```python
from ..models.ensemble import EnsembleStacker, create_xgboost_model, MetaLearnerType

# Create base models
base_models = [
    create_xgboost_model(n_estimators=100, max_depth=6),
    create_xgboost_model(n_estimators=150, max_depth=8),
]

# Create stacker
stacker = EnsembleStacker(
    base_models=base_models,
    meta_learner_type=MetaLearnerType.RIDGE
)

# Train
stacker.fit(X_train, y_train)
predictions = stacker.predict(X_test)
```

### 6. (Optional) Enable Correlation Regime Detection

In `main.py` trading loop, when calling `detect_regime()`, pass returns DataFrame:

```python
# Prepare returns DataFrame for correlation regime detection
returns_df = pl.DataFrame({
    symbol: df["close"].pct_change().drop_nulls()
    for symbol in config.data.symbols
})

# Use in confluence layer (requires returns_df parameter)
# Currently uses basic regime detection without returns_df
```

## 🎯 Quick Start - Minimum Required Steps

1. **Install dependencies**:
   ```bash
   pip install mlfinlab shap evidently nannyml feast
   ```

2. **That's it!** The system will work with:
   - Execution cost modeling (automatic)
   - Enhanced validation with gap days (when specified)
   - Dashboard monitoring tab (automatic)
   - SHAP (when saving models with X_val)

## 📊 Current Default Behavior

- **ExecutionCostModel**: Created automatically, estimates costs on all orders
- **SHAP**: Calculated when saving models (if X_val provided)
- **IC Tracker**: Not running (metrics will show 0)
- **Drift Detector**: Not running (no drift alerts)
- **Ensemble**: Not used (single XGBoost model)
- **Correlation Regime**: Not used (basic regime detection active)

## 🔄 Integration Priority

**High Priority** (Improves existing features):
1. Install dependencies
2. Enable SHAP when training (just pass X_val)

**Medium Priority** (Adds new monitoring):
3. Enable IC tracking (5-10 min integration)
4. Enable drift detection (10-15 min integration)

**Low Priority** (Alternative approaches):
5. Use ensemble models (requires retraining)
6. Enable correlation regime (requires data preparation)

## 📚 Documentation

All new features are documented in:
- `docs/research/tools/` - Tool guides (MLFinLab, SHAP)
- `docs/research/methodologies/` - Methodology docs
- `docs/research/papers/` - Research paper summaries
