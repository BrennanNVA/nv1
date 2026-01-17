"""Institutional-grade backtest validation and walk-forward optimization.

Research-backed implementation based on:
- "The Deflated Sharpe Ratio" (Bailey & Lopez de Prado, 2014)
- "Probability of Backtest Overfitting" (Bailey et al., 2015)
- "Walk-Forward Optimization" (Robert Pardo)
- "Advances in Financial Machine Learning" (Marcos Lopez de Prado)

Key methodology:
- Corrects for selection bias and multiple testing
- Detects overfitting via Combinatorially Symmetric Cross-Validation (CSCV)
- Implements rolling and expanding walk-forward windows
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Statistical validation results."""

    sharpe_ratio: float
    probabilistic_sharpe_ratio: float
    deflated_sharpe_ratio: float
    max_drawdown: float
    pbo: float  # Probability of Backtest Overfitting
    num_trials: int
    is_valid: bool


class BacktestValidator:
    """Statistical validator for trading strategies and models."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Initialize validator.

        Args:
            risk_free_rate: Annualized risk-free rate
        """
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(
        self, returns: Union[pl.Series, np.ndarray], annualization_factor: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Args:
            returns: Series of periodic returns
            annualization_factor: Periods per year (252 for daily)

        Returns:
            Annualized Sharpe Ratio
        """
        if isinstance(returns, pl.Series):
            returns = returns.to_numpy()

        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return 0.0

        return (mean_ret / std_ret) * np.sqrt(annualization_factor)

    def calculate_psr(
        self,
        returns: Union[pl.Series, np.ndarray],
        benchmark_sharpe: float = 0.0,
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).

        Corrects for non-normality and short track records.

        Args:
            returns: Series of periodic returns
            benchmark_sharpe: Target annualized Sharpe Ratio
            annualization_factor: Periods per year

        Returns:
            Probability that SR > benchmark_sharpe (0 to 1)
        """
        if isinstance(returns, pl.Series):
            returns = returns.to_numpy()

        n = len(returns)
        if n < 2:
            return 0.0

        sr = self.calculate_sharpe_ratio(returns, annualization_factor)

        # Calculate skewness and kurtosis
        skew = self._calculate_skew(returns)
        kurt = self._calculate_kurtosis(returns)

        # Standard error of SR
        std_err = np.sqrt(
            (
                1
                + (1 + 0.5 * skew**2) * (sr**2 / annualization_factor)
                - skew * sr / np.sqrt(annualization_factor)
                + (kurt - 3) / 4 * (sr**2 / annualization_factor)
            )
            / (n - 1)
        )

        # PSR calculation
        z_stat = (sr - benchmark_sharpe) / (std_err * np.sqrt(annualization_factor))
        return float(norm.cdf(z_stat))

    def calculate_dsr(
        self,
        returns: Union[pl.Series, np.ndarray],
        all_trials_sr: list[float],
        annualization_factor: int = 252,
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR).

        Corrects for selection bias (multiple testing).

        Args:
            returns: Series of periodic returns for the selected strategy
            all_trials_sr: Annualized Sharpe Ratios of all strategies tested
            annualization_factor: Periods per year

        Returns:
            Deflated Sharpe Ratio (probability SR > 0 after correction)
        """
        if not all_trials_sr:
            return self.calculate_psr(returns, 0.0, annualization_factor)

        n_trials = len(all_trials_sr)

        # Expected maximum SR under null hypothesis
        # Using Euler-Mascheroni constant approximation for expected max of N Gaussian variables
        em_const = 0.5772156649
        mean_sr = np.mean(all_trials_sr)
        std_sr = np.std(all_trials_sr)

        # Expected max SR (benchmark)
        exp_max_sr = mean_sr + std_sr * (
            (1 - em_const) * norm.ppf(1 - 1 / n_trials)
            + em_const * norm.ppf(1 - 1 / (n_trials * np.e))
        )

        return self.calculate_psr(returns, exp_max_sr, annualization_factor)

    def calculate_pbo(self, trial_returns_matrix: np.ndarray, n_partitions: int = 16) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO) via CSCV.

        Args:
            trial_returns_matrix: Matrix of size (T x N) where T is time and N is number of trials
            n_partitions: Number of combinations for CSCV (must be even)

        Returns:
            Probability of overfitting (0 to 1)
        """
        t, n = trial_returns_matrix.shape
        if n < 2 or t < n_partitions:
            return 0.0

        # Implementation of Combinatorially Symmetric Cross-Validation
        # 1. Partition rows into S segments
        partition_size = t // n_partitions
        segments = []
        for i in range(n_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < n_partitions - 1 else t
            segments.append(trial_returns_matrix[start:end, :])

        # 2. Form combinations of segments (using a simplified approach for speed)
        # We use a random sampling of combinations if n_partitions is large
        overfit_count = 0
        total_combinations = 0

        # To keep it efficient, we'll use a subset of all possible combinations
        # for CSCV if n_partitions is large. For institutional grade, we want at least 1000 trials.
        import itertools
        import random

        all_indices = list(range(n_partitions))
        k = n_partitions // 2

        # Limit combinations to 1000 for performance
        combos = list(itertools.combinations(all_indices, k))
        if len(combos) > 1000:
            combos = random.sample(combos, 1000)

        for train_indices in combos:
            test_indices = [i for i in all_indices if i not in train_indices]

            # Combine segments
            train_set = np.vstack([segments[i] for i in train_indices])
            test_set = np.vstack([segments[i] for i in test_indices])

            # Calculate SR for all trials in training set
            train_sr = np.mean(train_set, axis=0) / np.std(train_set, axis=0, ddof=1)
            best_trial_idx = np.argmax(train_sr)

            # Calculate SR for all trials in test set
            test_sr = np.mean(test_set, axis=0) / np.std(test_set, axis=0, ddof=1)

            # Rank of the "best" training trial in the test set
            # If rank < N/2, it overfits
            rank = np.sum(test_sr < test_sr[best_trial_idx]) / n
            if rank < 0.5:
                overfit_count += 1

            total_combinations += 1

        return overfit_count / total_combinations if total_combinations > 0 else 0.0

    def _calculate_skew(self, x: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(x)
        mu = np.mean(x)
        sigma = np.std(x, ddof=1)
        if sigma == 0:
            return 0.0
        return (np.sum((x - mu) ** 3) / n) / sigma**3

    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate sample kurtosis."""
        n = len(x)
        mu = np.mean(x)
        sigma = np.std(x, ddof=1)
        if sigma == 0:
            return 3.0
        return (np.sum((x - mu) ** 4) / n) / sigma**4


class WalkForwardOptimizer:
    """Implements Walk-Forward Optimization (WFO) with gap days and multi-horizon support."""

    def __init__(
        self,
        is_window_years: int = 2,
        oos_window_months: int = 6,
        step_months: int = 3,
        expanding: bool = False,
        gap_days: int = 0,  # Gap days between train/test
        horizons: Optional[list[int]] = None,  # Prediction horizons in days
    ) -> None:
        """
        Initialize WFO.

        Args:
            is_window_years: Length of in-sample training window
            oos_window_months: Length of out-of-sample validation window
            step_months: Step size for rolling windows
            expanding: If True, IS window grows from start
            gap_days: Gap days between training and test sets (prevents lookahead bias)
            horizons: List of prediction horizons in days (e.g., [1, 5, 20])
        """
        self.is_window_days = is_window_years * 252
        self.oos_window_days = oos_window_months * 21
        self.step_days = step_months * 21
        self.expanding = expanding
        self.gap_days = gap_days
        self.horizons = horizons or [1]  # Default to single horizon

    def generate_windows(self, df: pl.DataFrame) -> list[dict[str, Any]]:
        """
        Generate IS/OOS window indices with gap days.

        Args:
            df: DataFrame with datetime index

        Returns:
            List of window dictionaries with 'is_start', 'is_end', 'oos_start', 'oos_end', 'horizons'
        """
        n = len(df)
        windows = []

        current_oos_start = self.is_window_days

        while current_oos_start + self.gap_days + self.oos_window_days <= n:
            is_start = 0 if self.expanding else current_oos_start - self.is_window_days
            is_end = current_oos_start
            oos_start = current_oos_start + self.gap_days  # Add gap days
            oos_end = oos_start + self.oos_window_days

            # Ensure we have enough data for all horizons
            max_horizon = max(self.horizons) if self.horizons else 1
            if oos_end + max_horizon > n:
                break

            windows.append(
                {
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "gap_days": self.gap_days,
                    "horizons": self.horizons,
                    "is_start_dt": df["timestamp"][is_start] if "timestamp" in df.columns else None,
                    "is_end_dt": df["timestamp"][is_end - 1] if "timestamp" in df.columns else None,
                    "oos_start_dt": (
                        df["timestamp"][oos_start] if "timestamp" in df.columns else None
                    ),
                    "oos_end_dt": (
                        df["timestamp"][oos_end - 1] if "timestamp" in df.columns else None
                    ),
                }
            )

            current_oos_start += self.step_days

        logger.info(
            f"Generated {len(windows)} walk-forward windows "
            f"(gap_days={self.gap_days}, horizons={self.horizons})"
        )
        return windows

    def create_time_series_cv(
        self,
        n_splits: int = 5,
        gap: Optional[int] = None,
        test_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
    ) -> TimeSeriesSplit:
        """
        Create TimeSeriesSplit with gap days and multi-horizon support.

        Args:
            n_splits: Number of splits
            gap: Gap days between train/test (uses self.gap_days if None)
            test_size: Size of test set
            max_train_size: Maximum size of training set

        Returns:
            TimeSeriesSplit object
        """
        gap_days = gap if gap is not None else self.gap_days

        return TimeSeriesSplit(
            n_splits=n_splits, gap=gap_days, test_size=test_size, max_train_size=max_train_size
        )

    def validate_multi_horizon(
        self,
        predictions: dict[int, np.ndarray],
        actuals: dict[int, np.ndarray],
    ) -> dict[int, dict[str, float]]:
        """
        Validate predictions at multiple horizons.

        Args:
            predictions: Dictionary mapping horizon to prediction array
            actuals: Dictionary mapping horizon to actual array

        Returns:
            Dictionary mapping horizon to validation metrics
        """
        results = {}

        for horizon in self.horizons:
            if horizon not in predictions or horizon not in actuals:
                logger.warning(f"Missing predictions or actuals for horizon {horizon}")
                continue

            pred = predictions[horizon]
            actual = actuals[horizon]

            if len(pred) != len(actual):
                logger.warning(f"Length mismatch for horizon {horizon}")
                continue

            # Calculate metrics
            mse = np.mean((pred - actual) ** 2)
            mae = np.mean(np.abs(pred - actual))
            rmse = np.sqrt(mse)

            # IC (Information Coefficient) - correlation
            if np.std(pred) > 0 and np.std(actual) > 0:
                ic = np.corrcoef(pred, actual)[0, 1]
            else:
                ic = 0.0

            results[horizon] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "ic": float(ic) if not np.isnan(ic) else 0.0,
            }

        return results
