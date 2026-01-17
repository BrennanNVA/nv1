"""Information Coefficient (IC) decay monitoring and signal half-life calculation.

Research-backed implementation based on:
- "Information Coefficient Decay" (Lopez de Prado, AFML)
- Alpha decay typically 5.6% annually in US markets, 9.9% in Europe
- Signal half-life estimation using AR(1) / Ornstein-Uhlenbeck processes

Key features:
- IC calculation at multiple forward horizons (1-day, 5-day, 20-day)
- Rolling IC monitoring over time
- Signal half-life calculation
- Automated alerts when IC drops below threshold
- Integration with Prometheus metrics
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats

from ..core.metrics import get_metrics

logger = logging.getLogger(__name__)


@dataclass
class ICStats:
    """IC statistics for a signal."""

    mean_ic: float
    std_ic: float
    ic_ir: float  # IC Information Ratio (mean / std)
    positive_ratio: float  # Percentage of positive IC
    half_life_days: Optional[float] = None
    decay_rate: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SignalIC:
    """IC tracking for a single signal."""

    signal_name: str
    horizons: list[int]  # Forward horizons in days
    ic_history: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    predictions: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    actuals: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    timestamps: list[datetime] = field(default_factory=list)

    def add_observation(
        self, horizon: int, prediction: float, actual: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Add an observation for IC calculation."""
        if horizon not in self.horizons:
            logger.warning(f"Horizon {horizon} not in tracked horizons")
            return

        self.predictions[horizon].append(prediction)
        self.actuals[horizon].append(actual)
        self.timestamps.append(timestamp or datetime.now())

        # Keep limited history (last 1000 observations)
        max_history = 1000
        if len(self.predictions[horizon]) > max_history:
            self.predictions[horizon] = self.predictions[horizon][-max_history:]
            self.actuals[horizon] = self.actuals[horizon][-max_history:]
            self.timestamps = self.timestamps[-max_history:]

    def calculate_ic(self, horizon: int, window: Optional[int] = None) -> float:
        """
        Calculate IC for a specific horizon.

        Args:
            horizon: Forward horizon in days
            window: Optional rolling window (None = all history)

        Returns:
            IC (correlation coefficient)
        """
        if horizon not in self.predictions or not self.predictions[horizon]:
            return 0.0

        preds = np.array(self.predictions[horizon])
        actuals = np.array(self.actuals[horizon])

        if window:
            preds = preds[-window:]
            actuals = actuals[-window:]

        if len(preds) < 2:
            return 0.0

        # Calculate Pearson correlation
        if np.std(preds) == 0 or np.std(actuals) == 0:
            return 0.0

        ic, _ = stats.pearsonr(preds, actuals)
        return float(ic) if not np.isnan(ic) else 0.0

    def calculate_rolling_ic(self, horizon: int, window: int = 20) -> list[float]:
        """
        Calculate rolling IC over time.

        Args:
            horizon: Forward horizon in days
            window: Rolling window size

        Returns:
            List of rolling IC values
        """
        if horizon not in self.predictions or not self.predictions[horizon]:
            return []

        preds = np.array(self.predictions[horizon])
        actuals = np.array(self.actuals[horizon])

        if len(preds) < window:
            return []

        rolling_ics = []
        for i in range(window, len(preds) + 1):
            window_preds = preds[i - window : i]
            window_actuals = actuals[i - window : i]

            if np.std(window_preds) == 0 or np.std(window_actuals) == 0:
                rolling_ics.append(0.0)
                continue

            ic, _ = stats.pearsonr(window_preds, window_actuals)
            rolling_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        return rolling_ics


class ICTracker:
    """Track Information Coefficient (IC) decay for multiple signals and horizons.

    Implements:
    - IC calculation at multiple forward horizons
    - Rolling IC monitoring
    - Signal half-life calculation using AR(1) / Ornstein-Uhlenbeck
    - Automated alerts when IC drops below threshold
    - Integration with Prometheus metrics
    """

    def __init__(
        self,
        horizons: list[int] = None,
        alert_threshold: float = 0.05,
        rolling_window: int = 20,
    ) -> None:
        """
        Initialize IC tracker.

        Args:
            horizons: List of forward horizons in days (default: [1, 5, 20])
            alert_threshold: IC threshold for alerts (default: 0.05)
            rolling_window: Window size for rolling IC (default: 20)
        """
        self.horizons = horizons or [1, 5, 20]
        self.alert_threshold = alert_threshold
        self.rolling_window = rolling_window

        # Track IC for each signal
        self.signals: dict[str, SignalIC] = {}

        # Metrics collector
        self.metrics = get_metrics()

        logger.info(
            f"ICTracker initialized: horizons={self.horizons}, "
            f"alert_threshold={self.alert_threshold}"
        )

    def register_signal(self, signal_name: str) -> None:
        """Register a new signal for IC tracking."""
        if signal_name not in self.signals:
            self.signals[signal_name] = SignalIC(signal_name=signal_name, horizons=self.horizons)
            logger.info(f"Registered signal for IC tracking: {signal_name}")

    def record_prediction(
        self,
        signal_name: str,
        horizon: int,
        prediction: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a prediction (to be matched with actual later).

        Args:
            signal_name: Name of the signal
            horizon: Forward horizon in days
            prediction: Predicted return/signal value
            timestamp: Timestamp of prediction
        """
        if signal_name not in self.signals:
            self.register_signal(signal_name)

        self.signals[signal_name].add_observation(
            horizon=horizon,
            prediction=prediction,
            actual=0.0,  # Placeholder, will be updated
            timestamp=timestamp,
        )

    def record_actual(
        self, signal_name: str, horizon: int, actual: float, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record actual return (matches with most recent prediction).

        Args:
            signal_name: Name of the signal
            horizon: Forward horizon in days
            actual: Actual return
            timestamp: Timestamp of actual return
        """
        if signal_name not in self.signals:
            logger.warning(f"Signal {signal_name} not registered")
            return

        signal = self.signals[signal_name]

        # Find matching prediction (most recent for this horizon)
        if horizon not in signal.predictions or not signal.predictions[horizon]:
            logger.warning(f"No predictions found for {signal_name} at horizon {horizon}")
            return

        # Update the most recent actual (if it was a placeholder)
        if signal.actuals[horizon] and signal.actuals[horizon][-1] == 0.0:
            signal.actuals[horizon][-1] = actual
        else:
            # Add new observation
            signal.add_observation(
                horizon=horizon,
                prediction=signal.predictions[horizon][-1] if signal.predictions[horizon] else 0.0,
                actual=actual,
                timestamp=timestamp,
            )

        # Calculate IC and update metrics
        ic = signal.calculate_ic(horizon, window=self.rolling_window)

        # Update Prometheus metrics
        self.metrics.set_gauge(
            f"ic_{signal_name}_horizon_{horizon}",
            ic,
            {"signal": signal_name, "horizon": str(horizon)},
        )

        # Check for alerts
        if ic < self.alert_threshold:
            logger.warning(
                f"IC alert: {signal_name} at horizon {horizon}d has IC={ic:.4f} "
                f"(threshold={self.alert_threshold})"
            )
            self.metrics.increment(
                "ic_alert_total", labels={"signal": signal_name, "horizon": str(horizon)}
            )

    def get_ic_stats(self, signal_name: str, horizon: int, window: Optional[int] = None) -> ICStats:
        """
        Get IC statistics for a signal and horizon.

        Args:
            signal_name: Name of the signal
            horizon: Forward horizon in days
            window: Optional rolling window

        Returns:
            ICStats object with statistics
        """
        if signal_name not in self.signals:
            return ICStats(0.0, 0.0, 0.0, 0.0)

        signal = self.signals[signal_name]
        rolling_ic = signal.calculate_rolling_ic(horizon, window or self.rolling_window)

        if not rolling_ic:
            return ICStats(0.0, 0.0, 0.0, 0.0)

        mean_ic = np.mean(rolling_ic)
        std_ic = np.std(rolling_ic)
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        positive_ratio = np.sum(np.array(rolling_ic) > 0) / len(rolling_ic)

        # Calculate half-life
        half_life = self._calculate_half_life(rolling_ic)
        decay_rate = self._calculate_decay_rate(rolling_ic)

        return ICStats(
            mean_ic=float(mean_ic),
            std_ic=float(std_ic),
            ic_ir=float(ic_ir),
            positive_ratio=float(positive_ratio),
            half_life_days=half_life,
            decay_rate=decay_rate,
        )

    def _calculate_half_life(self, ic_series: list[float]) -> Optional[float]:
        """
        Calculate signal half-life using AR(1) / Ornstein-Uhlenbeck process.

        Half-life = -log(2) / log(autocorrelation)

        Args:
            ic_series: Series of IC values

        Returns:
            Half-life in days (or None if cannot be calculated)
        """
        if len(ic_series) < 10:
            return None

        ic_array = np.array(ic_series)

        # Calculate first-order autocorrelation
        if len(ic_array) < 2:
            return None

        # Remove mean
        ic_centered = ic_array - np.mean(ic_array)

        if np.std(ic_centered) == 0:
            return None

        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(ic_centered[:-1], ic_centered[1:])[0, 1]

        if np.isnan(autocorr) or autocorr <= 0:
            return None

        # Half-life = -log(2) / log(autocorr)
        # But we need to convert to days based on observation frequency
        # Assuming daily observations
        half_life = -np.log(2) / np.log(autocorr)

        return float(half_life) if not np.isnan(half_life) else None

    def _calculate_decay_rate(self, ic_series: list[float]) -> Optional[float]:
        """
        Calculate IC decay rate (annualized).

        Args:
            ic_series: Series of IC values

        Returns:
            Annualized decay rate (or None if cannot be calculated)
        """
        if len(ic_series) < 20:
            return None

        # Fit linear trend to IC series
        x = np.arange(len(ic_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ic_series)

        # Annualize decay rate (assuming daily observations, 252 trading days)
        annual_decay = slope * 252

        return float(annual_decay) if not np.isnan(annual_decay) else None

    def get_all_ic_stats(self) -> dict[str, dict[int, ICStats]]:
        """
        Get IC statistics for all signals and horizons.

        Returns:
            Nested dictionary: {signal_name: {horizon: ICStats}}
        """
        all_stats = {}

        for signal_name, signal in self.signals.items():
            all_stats[signal_name] = {}
            for horizon in self.horizons:
                all_stats[signal_name][horizon] = self.get_ic_stats(signal_name, horizon)

        return all_stats

    def get_rolling_ic(
        self, signal_name: str, horizon: int, window: Optional[int] = None
    ) -> list[float]:
        """
        Get rolling IC series for a signal and horizon.

        Args:
            signal_name: Name of the signal
            horizon: Forward horizon in days
            window: Rolling window size

        Returns:
            List of rolling IC values
        """
        if signal_name not in self.signals:
            return []

        return self.signals[signal_name].calculate_rolling_ic(
            horizon, window or self.rolling_window
        )
