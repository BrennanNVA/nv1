# MLFinLab Free Alternative

## Overview

MLFinLab is a commercial library from Hudson & Thames. This document describes the **free, open-source alternative** implemented in Nova Aetus that provides the same key functionality.

## Implementation

All MLFinLab features are implemented in `src/nova/features/finml_utils.py`:

### ✅ Features Implemented

1. **Purged k-Fold Cross-Validation** (`PurgedKFold`)
   - Prevents lookahead bias
   - Embargo period support
   - Label window purging

2. **Combinatorial Purged k-Fold** (`CombinatorialPurgedKFold`)
   - CSCV for PBO calculation
   - More robust than simple splits

3. **Optimal d Selection** (`find_optimal_d`)
   - Finds best fractional differentiation parameter
   - Uses ADF test for stationarity
   - Maximizes memory preservation

4. **Triple-Barrier Labeling** (`triple_barrier_labels`)
   - Profit-taking barrier
   - Stop-loss barrier
   - Minimum return threshold

### Already Available

5. **Fractional Differentiation** - Already implemented in `src/nova/features/technical.py`
   - Custom `fractional_diff()` method
   - Fixed-width window implementation
   - Based on Lopez de Prado's methodology

## Usage

### Purged k-Fold Cross-Validation

```python
from nova.features.finml_utils import PurgedKFold

# Create purged CV
cv = PurgedKFold(
    n_splits=5,
    t1=label_end_times,  # Label end times
    pct_embargo=0.01  # 1% embargo
)

# Use in cross-validation
for train_idx, test_idx in cv.split(X, y, groups=label_end_times):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### Optimal d Selection

```python
from nova.features.finml_utils import find_optimal_d

# Find optimal d for fractional differentiation
optimal_d, results = find_optimal_d(
    series=price_series,
    max_d=1.0,
    step=0.05,
    adf_pvalue_threshold=0.05
)

print(f"Optimal d: {optimal_d}")
```

### Triple-Barrier Labeling

```python
from nova.features.finml_utils import triple_barrier_labels

labels = triple_barrier_labels(
    prices=price_series,
    t_events=volatility_events,
    pt=0.02,  # 2% profit-taking
    sl=0.01,  # 1% stop-loss
    min_ret=0.005  # 0.5% minimum return
)
```

## Comparison with MLFinLab

| Feature | MLFinLab | Free Alternative | Status |
|---------|----------|------------------|--------|
| Purged k-Fold | ✅ | ✅ | Implemented |
| Combinatorial Purged k-Fold | ✅ | ✅ | Implemented |
| Fractional Differentiation | ✅ | ✅ | Already in codebase |
| Optimal d Selection | ✅ | ✅ | Implemented |
| Triple-Barrier Labeling | ✅ | ✅ | Implemented |
| Plotting utilities | ✅ | ⚠️ | Manual plotting |
| Commercial Support | ✅ | ❌ | N/A |

## Advantages of Free Alternative

- **No cost**: Completely free and open-source
- **Transparent**: Full source code available
- **Customizable**: Easy to modify for your needs
- **Lightweight**: Only depends on scikit-learn, scipy, statsmodels

## Limitations

- No commercial support
- May need to implement additional utilities manually
- Plotting utilities need to be created separately (if needed)

## Integration

The free alternative is already integrated into:
- `src/nova/models/trainer.py` - Purged CV support
- `src/nova/models/training_pipeline.py` - Purged CV usage
- `src/nova/features/technical.py` - Fractional differentiation

## References

- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
- Implementation based on open-source research and methodologies
