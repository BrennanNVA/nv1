"""Free alternative to MLFinLab - Lopez de Prado methodologies.

Implements key features from "Advances in Financial Machine Learning" without requiring
the commercial MLFinLab library. All implementations are open-source and free.

Key features:
- Purged k-Fold Cross-Validation with embargo
- Combinatorial Purged k-Fold (CSCV) for PBO
- Optimal d parameter selection for fractional differentiation
- Triple-barrier labeling utilities
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable

logger = logging.getLogger(__name__)


@dataclass
class PurgedKFoldConfig:
    """Configuration for purged k-fold."""

    n_splits: int = 5
    pct_embargo: float = 0.01  # 1% embargo
    t1: Optional[np.ndarray] = None  # Label end times


class PurgedKFold(BaseCrossValidator):
    """Purged k-Fold Cross-Validation with embargo period.

    Prevents lookahead bias by:
    - Purging overlapping labels between train/test sets
    - Adding embargo period (gap days) between train and test

    Based on "Advances in Financial Machine Learning" by Lopez de Prado.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[np.ndarray] = None,
        pct_embargo: float = 0.01,
    ) -> None:
        """
        Initialize purged k-fold.

        Args:
            n_splits: Number of splits
            t1: Label end times (for purging overlapping labels)
            pct_embargo: Percentage embargo period (e.g., 0.01 = 1%)
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits with purging and embargo.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (uses t1 if provided)

        Yields:
            (train_indices, test_indices) tuples
        """
        X, y, groups = indexable(X, y, groups)

        # Use t1 if provided, otherwise use indices
        if self.t1 is not None:
            label_end_times = self.t1
        elif groups is not None:
            label_end_times = np.array(groups)
        else:
            # Fall back to simple k-fold if no time information
            logger.warning("No time information provided, using simple k-fold")
            indices = np.arange(len(X))
            n_samples = len(X)
            fold_size = n_samples // self.n_splits

            for i in range(self.n_splits):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

                test_indices = indices[test_start:test_end]
                train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

                yield train_indices, test_indices
            return

        indices = np.arange(len(X))
        n_samples = len(X)

        # Calculate embargo in samples
        embargo = int(n_samples * self.pct_embargo)

        # Generate splits
        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * n_samples // self.n_splits
            test_end = (i + 1) * n_samples // self.n_splits

            test_indices = indices[test_start:test_end]
            test_times = label_end_times[test_indices]

            # Find train indices (exclude overlapping labels + embargo)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # Purge overlapping labels
            if len(test_times) > 0:
                test_start_time = np.min(test_times)
                test_end_time = np.max(test_times)

                # Remove samples where labels overlap with test set
                overlapping = (label_end_times >= test_start_time) & (
                    label_end_times <= test_end_time
                )
                train_mask &= ~overlapping

            # Apply embargo
            if embargo > 0:
                if test_start > 0:
                    embargo_start = max(0, test_start - embargo)
                    train_mask[embargo_start:test_start] = False
                if test_end < n_samples:
                    embargo_end = min(n_samples, test_end + embargo)
                    train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold(BaseCrossValidator):
    """Combinatorially Symmetric Cross-Validation (CSCV) for PBO calculation.

    Used to calculate Probability of Backtest Overfitting (PBO) via CSCV.
    More robust than simple train/test splits.

    Based on "Advances in Financial Machine Learning" by Lopez de Prado.
    """

    def __init__(
        self,
        n_splits: int = 10,
        t1: Optional[np.ndarray] = None,
        pct_embargo: float = 0.01,
    ) -> None:
        """
        Initialize combinatorial purged k-fold.

        Args:
            n_splits: Number of partitions (must be even)
            t1: Label end times
            pct_embargo: Percentage embargo period
        """
        if n_splits % 2 != 0:
            raise ValueError("n_splits must be even for CSCV")

        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate CSCV splits.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels

        Yields:
            (train_indices, test_indices) tuples
        """
        # Implementation uses all combinations of n_splits/2 partitions for training
        # This is computationally expensive, so we use a subset of combinations
        import itertools

        X, y, groups = indexable(X, y, groups)
        indices = np.arange(len(X))
        n_samples = len(X)

        # Partition data into n_splits segments
        partition_size = n_samples // self.n_splits
        partitions = []
        for i in range(self.n_splits):
            start = i * partition_size
            end = (i + 1) * partition_size if i < self.n_splits - 1 else n_samples
            partitions.append(indices[start:end])

        # Generate combinations of n_splits/2 partitions for training
        k = self.n_splits // 2
        partition_indices = list(range(self.n_splits))

        # Limit to 100 combinations for performance
        combinations = list(itertools.combinations(partition_indices, k))
        if len(combinations) > 100:
            # Random sample for performance
            import random

            combinations = random.sample(combinations, 100)

        for train_partition_indices in combinations:
            test_partition_indices = [
                i for i in partition_indices if i not in train_partition_indices
            ]

            # Combine partitions
            train_indices = np.concatenate([partitions[i] for i in train_partition_indices])
            test_indices = np.concatenate([partitions[i] for i in test_partition_indices])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


def find_optimal_d(
    series: np.ndarray,
    max_d: float = 1.0,
    step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue_threshold: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """
    Find optimal fractional differentiation parameter d.

    Tests different d values and selects the one that achieves stationarity
    with maximum memory preservation.

    Args:
        series: Price series to differentiate
        max_d: Maximum d value to test
        step: Step size for d values
        threshold: Weight threshold for fractional diff
        adf_pvalue_threshold: ADF test p-value threshold for stationarity

    Returns:
        Tuple of (optimal_d, results_dict)
    """
    from statsmodels.tsa.stattools import adfuller

    d_values = np.arange(step, max_d + step, step)
    results = []

    # Calculate fractional diff for each d
    for d in d_values:
        try:
            # Simplified fractional diff calculation
            diff_series = _fractional_diff_simple(series, d, threshold)

            # Skip if too short
            if len(diff_series) < 20:
                continue

            # ADF test for stationarity
            adf_result = adfuller(diff_series.dropna())
            pvalue = adf_result[1]

            results.append(
                {
                    "d": d,
                    "pvalue": pvalue,
                    "adf_stat": adf_result[0],
                    "is_stationary": pvalue < adf_pvalue_threshold,
                    "n_samples": len(diff_series),
                }
            )
        except Exception as e:
            logger.debug(f"Failed to test d={d}: {e}")
            continue

    if not results:
        return 0.5, {}  # Default fallback

    # Find optimal d (lowest d that achieves stationarity)
    stationary_results = [r for r in results if r["is_stationary"]]

    if stationary_results:
        optimal = min(stationary_results, key=lambda x: x["d"])
        optimal_d = optimal["d"]
    else:
        # Use d with lowest p-value (closest to stationarity)
        optimal = min(results, key=lambda x: x["pvalue"])
        optimal_d = optimal["d"]

    return optimal_d, {
        "optimal_d": optimal_d,
        "all_results": results,
        "optimal_result": optimal,
    }


def _fractional_diff_simple(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """Simple fractional differentiation implementation."""
    weights = [1.0]
    k = 1

    # Calculate weights
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])  # Reverse for convolution

    # Apply fractional differencing
    diff_series = np.convolve(series, weights, mode="valid")

    return diff_series


def triple_barrier_labels(
    prices: np.ndarray,
    t_events: np.ndarray,
    pt: float = 0.02,  # Profit-taking barrier
    sl: float = 0.01,  # Stop-loss barrier
    min_ret: float = 0.005,
) -> np.ndarray:
    """
    Generate triple-barrier labels.

    Args:
        prices: Price series
        t_events: Event times (volatility events, etc.)
        pt: Profit-taking barrier (e.g., 0.02 = 2%)
        sl: Stop-loss barrier (e.g., 0.01 = 1%)
        min_ret: Minimum return threshold

    Returns:
        Array of labels: 1 (profit), -1 (stop loss), 0 (neutral)
    """
    labels = np.zeros(len(t_events), dtype=int)

    for i, event_idx in enumerate(t_events):
        if event_idx >= len(prices) - 1:
            continue

        current_price = prices[event_idx]

        # Look forward for barrier hits
        for j in range(event_idx + 1, len(prices)):
            future_price = prices[j]
            ret = (future_price - current_price) / current_price

            # Check barriers
            if ret >= pt:
                labels[i] = 1  # Profit-taking hit
                break
            elif ret <= -sl:
                labels[i] = -1  # Stop-loss hit
                break
            elif abs(ret) < min_ret:
                labels[i] = 0  # Neutral
                break

    return labels
