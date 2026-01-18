# MLFinLab Integration Guide

## Overview

MLFinLab is the official implementation of Marcos Lopez de Prado's "Advances in Financial Machine Learning" (AFML). This library provides battle-tested implementations of critical financial ML methodologies.

## Why MLFinLab?

- **Battle-tested**: Official implementation from Hudson & Thames
- **Replaces custom code**: Our custom fractional differentiation can be replaced with `frac_diff_ffd()`
- **Proper validation**: Includes `PurgedKFold` and `CombinatorialPurgedKFold` for correct cross-validation
- **Research-backed**: Implements methodologies from the AFML book

## Key Features Used in Nova Aetus

### 1. Fixed-Width Fractional Differentiation (`frac_diff_ffd`)

**Replaces**: Custom fractional differentiation implementation

**Usage**:
```python
from mlfinlab.fractional_differentiation import frac_diff_ffd

# Calculate fractional differentiation with fixed width
diff_series = frac_diff_ffd(
    series=price_series,
    d=0.5,  # Differentiation order
    thresh=1e-5  # Threshold for stationarity
)
```

**Benefits**:
- Proper stationarity testing
- Optimal `d` parameter selection via `plot_min_ffd()`
- More robust than custom implementation

### 2. Purged k-Fold Cross-Validation (`PurgedKFold`)

**Replaces**: Basic walk-forward validation

**Usage**:
```python
from mlfinlab.cross_validation import PurgedKFold

# Create purged CV with embargo
cv = PurgedKFold(
    n_splits=5,
    t1=label_times,  # Label start times
    pct_embargo=0.01  # 1% embargo period
)

for train_idx, test_idx in cv.split(X, y, groups=label_times):
    # Train/test split with proper purging
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**Benefits**:
- Prevents lookahead bias
- Proper embargo period (gap days)
- Label windows don't overlap train/test sets

### 3. Combinatorial Purged k-Fold (`CombinatorialPurgedKFold`)

**Purpose**: Calculate Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation (CSCV)

**Usage**:
```python
from mlfinlab.cross_validation import CombinatorialPurgedKFold

# CSCV for PBO calculation
cv = CombinatorialPurgedKFold(
    n_splits=10,
    t1=label_times,
    pct_embargo=0.01
)

# Calculate PBO
pbo = calculate_pbo(cv, model_results)
```

**Benefits**:
- Detects overfitting via CSCV
- More robust than simple train/test split
- Used in our `BacktestValidator` for PBO calculation

### 4. Triple-Barrier Labeling

**Purpose**: Generate labels based on price barriers (upper, lower, vertical)

**Usage**:
```python
from mlfinlab.labeling import get_events, get_bins

# Define barriers
events = get_events(
    close=price_series,
    t_events=volatility_events,
    pt=0.02,  # Profit-taking barrier (2%)
    sl=0.01,  # Stop-loss barrier (1%)
    min_ret=0.005,  # Minimum return
    num_threads=1,
    t1=None  # Vertical barrier (time-based)
)

# Generate labels
labels = get_bins(events, close)
```

**Benefits**:
- More sophisticated than simple return-based labels
- Accounts for volatility
- Can be combined with NPMM labeling

### 5. Sample Weights for Imbalanced Data

**Purpose**: Weight samples based on time decay or other factors

**Usage**:
```python
from mlfinlab.sample_weights import get_weights_by_time_decay

# Time-decay weights (recent samples more important)
weights = get_weights_by_time_decay(
    t1=label_times,
    decay=0.5,  # Decay factor
    max_weight=1.0
)
```

**Benefits**:
- Handles imbalanced data
- Time-decay for regime changes
- Improves model training

## Integration Points

### In `src/nova/features/technical.py`

Replace custom fractional differentiation:
```python
from mlfinlab.fractional_differentiation import frac_diff_ffd

def calculate_fractional_diff(series: pl.Series, d: float = 0.5) -> pl.Series:
    """Calculate fractional differentiation using MLFinLab."""
    return frac_diff_ffd(series.to_numpy(), d=d)
```

### In `src/nova/models/training_pipeline.py`

Add purged CV:
```python
from mlfinlab.cross_validation import PurgedKFold

# Use PurgedKFold instead of basic TimeSeriesSplit
cv = PurgedKFold(
    n_splits=5,
    t1=label_times,
    pct_embargo=0.01
)
```

### In `src/nova/models/validation.py`

Enhance PBO calculation:
```python
from mlfinlab.cross_validation import CombinatorialPurgedKFold

# Use CSCV for more robust PBO
cv = CombinatorialPurgedKFold(
    n_splits=10,
    t1=label_times,
    pct_embargo=0.01
)
```

## Optimal `d` Parameter Selection

Use `plot_min_ffd()` to find optimal differentiation order:

```python
from mlfinlab.fractional_differentiation import plot_min_ffd

# Plot to find optimal d
plot_min_ffd(
    series=price_series,
    max_d=1.0,
    step=0.05
)
```

## Installation

```bash
pip install mlfinlab
```

**Note**: MLFinLab may have dependencies on specific versions of pandas/numpy. Check compatibility.

## References

- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
- MLFinLab Documentation: https://mlfinlab.readthedocs.io/
- Hudson & Thames: https://hudsonthames.org/

## Impact

- **Replaces custom fractional diff**: More robust, battle-tested
- **Proper purged CV**: Prevents lookahead bias
- **Better validation**: CSCV for PBO calculation
- **Research-backed**: Implements AFML methodologies
