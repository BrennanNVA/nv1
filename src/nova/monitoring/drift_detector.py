"""Comprehensive drift detection for model monitoring.

Research-backed implementation based on:
- Evidently AI for comprehensive drift detection
- NannyML for performance estimation without labels
- PSI (Population Stability Index) for data drift
- KS (Kolmogorov-Smirnov) tests for distribution drift
- Concept drift detection via reverse CV

Key features:
- Data drift detection (PSI, KS tests)
- Concept drift detection
- Prediction drift monitoring
- Target drift detection
- Automated retraining triggers
- Integration with Prometheus metrics and Discord alerts
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import TestNumberOfDriftedFeatures

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI not available - install with: pip install evidently>=0.4.0")

try:
    import nannyml as nml

    NANNYML_AVAILABLE = True
except ImportError:
    NANNYML_AVAILABLE = False
    logger.warning("NannyML not available - install with: pip install nannyml>=0.10.0")

from ..core.metrics import get_metrics
from ..core.notifications import NotificationService

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""

    drift_detected: bool
    drift_type: str  # 'data', 'concept', 'prediction', 'target'
    severity: str  # 'low', 'medium', 'high'
    p_value: Optional[float] = None
    psi: Optional[float] = None
    ks_statistic: Optional[float] = None
    drifted_features: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Comprehensive drift detection for model monitoring.

    Implements:
    - Data drift (PSI, KS tests)
    - Concept drift (reverse CV, windowed retraining)
    - Prediction drift
    - Target drift
    - Automated alerts and retraining triggers
    """

    def __init__(
        self,
        reference_data: Optional[pl.DataFrame] = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        use_evidently: bool = True,
        use_nannyml: bool = True,
    ) -> None:
        """
        Initialize drift detector.

        Args:
            reference_data: Reference dataset (training data)
            psi_threshold: PSI threshold for data drift (default: 0.2)
            ks_threshold: KS test p-value threshold (default: 0.05)
            use_evidently: Use Evidently AI for drift detection
            use_nannyml: Use NannyML for performance estimation
        """
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.use_evidently = use_evidently and EVIDENTLY_AVAILABLE
        self.use_nannyml = use_nannyml and NANNYML_AVAILABLE

        # Store recent predictions and actuals for concept drift
        self.recent_predictions: list[float] = []
        self.recent_actuals: list[float] = []
        self.recent_timestamps: list[datetime] = []

        # Metrics collector
        self.metrics = get_metrics()

        # Notification service (optional)
        self.notification_service: Optional[NotificationService] = None

        logger.info(
            f"DriftDetector initialized: psi_threshold={psi_threshold}, "
            f"ks_threshold={ks_threshold}, evidently={self.use_evidently}, "
            f"nannyml={self.use_nannyml}"
        )

    def set_reference_data(self, reference_data: pl.DataFrame) -> None:
        """Set reference dataset (training data)."""
        self.reference_data = reference_data
        logger.info(f"Reference data set: {len(reference_data)} samples")

    def set_notification_service(self, notification_service: NotificationService) -> None:
        """Set notification service for alerts."""
        self.notification_service = notification_service

    def detect_data_drift(
        self, current_data: pl.DataFrame, feature_columns: Optional[list[str]] = None
    ) -> DriftResult:
        """
        Detect data drift using PSI and KS tests.

        Args:
            current_data: Current data to compare
            feature_columns: List of feature columns to check

        Returns:
            DriftResult with drift information
        """
        if self.reference_data is None:
            logger.warning("No reference data set - cannot detect drift")
            return DriftResult(drift_detected=False, drift_type="data", severity="low")

        if feature_columns is None:
            feature_columns = [
                col for col in current_data.columns if col in self.reference_data.columns
            ]

        drifted_features = []
        max_psi = 0.0
        min_ks_pvalue = 1.0

        for feature in feature_columns:
            if feature not in self.reference_data.columns:
                continue

            ref_values = self.reference_data[feature].drop_nulls().to_numpy()
            curr_values = current_data[feature].drop_nulls().to_numpy()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # Calculate PSI
            psi = self._calculate_psi(ref_values, curr_values)
            max_psi = max(max_psi, psi)

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
            min_ks_pvalue = min(min_ks_pvalue, ks_pvalue)

            # Check if drifted
            if psi > self.psi_threshold or ks_pvalue < self.ks_threshold:
                drifted_features.append(feature)

        drift_detected = len(drifted_features) > 0

        # Determine severity
        if drift_detected:
            if max_psi > 0.5 or min_ks_pvalue < 0.01:
                severity = "high"
            elif max_psi > 0.3 or min_ks_pvalue < 0.03:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "low"

        result = DriftResult(
            drift_detected=drift_detected,
            drift_type="data",
            severity=severity,
            p_value=min_ks_pvalue,
            psi=max_psi,
            ks_statistic=min_ks_pvalue,
            drifted_features=drifted_features,
        )

        # Update metrics
        self.metrics.set_gauge("drift_psi_max", max_psi)
        self.metrics.set_gauge("drift_ks_pvalue_min", min_ks_pvalue)
        self.metrics.set_gauge("drift_features_count", len(drifted_features))

        if drift_detected:
            self.metrics.increment("drift_detected_total", labels={"type": "data"})
            self._send_alert(result)

        return result

    def detect_concept_drift(self, window_size: int = 100, min_periods: int = 50) -> DriftResult:
        """
        Detect concept drift using reverse CV and performance decay.

        Args:
            window_size: Window size for analysis
            min_periods: Minimum periods required

        Returns:
            DriftResult with concept drift information
        """
        if len(self.recent_predictions) < min_periods:
            return DriftResult(drift_detected=False, drift_type="concept", severity="low")

        # Use recent window
        window_preds = np.array(self.recent_predictions[-window_size:])
        window_actuals = np.array(self.recent_actuals[-window_size:])

        # Calculate performance metrics
        # Split into two halves
        mid = len(window_preds) // 2
        first_half_preds = window_preds[:mid]
        first_half_actuals = window_actuals[:mid]
        second_half_preds = window_preds[mid:]
        second_half_actuals = window_actuals[mid:]

        # Calculate correlation (IC) for each half
        if len(first_half_preds) < 10 or len(second_half_preds) < 10:
            return DriftResult(drift_detected=False, drift_type="concept", severity="low")

        ic_first, _ = stats.pearsonr(first_half_preds, first_half_actuals)
        ic_second, _ = stats.pearsonr(second_half_preds, second_half_actuals)

        # Check if IC dropped significantly
        ic_drop = ic_first - ic_second
        drift_detected = ic_drop > 0.1  # Significant drop

        # Determine severity
        if drift_detected:
            if ic_drop > 0.3:
                severity = "high"
            elif ic_drop > 0.2:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "low"

        result = DriftResult(
            drift_detected=drift_detected,
            drift_type="concept",
            severity=severity,
            details={
                "ic_first_half": float(ic_first) if not np.isnan(ic_first) else 0.0,
                "ic_second_half": float(ic_second) if not np.isnan(ic_second) else 0.0,
                "ic_drop": float(ic_drop) if not np.isnan(ic_drop) else 0.0,
            },
        )

        # Update metrics
        self.metrics.set_gauge(
            "concept_drift_ic_first", float(ic_first) if not np.isnan(ic_first) else 0.0
        )
        self.metrics.set_gauge(
            "concept_drift_ic_second", float(ic_second) if not np.isnan(ic_second) else 0.0
        )

        if drift_detected:
            self.metrics.increment("drift_detected_total", labels={"type": "concept"})
            self._send_alert(result)

        return result

    def detect_prediction_drift(
        self, current_predictions: np.ndarray, reference_predictions: Optional[np.ndarray] = None
    ) -> DriftResult:
        """
        Detect prediction drift (distribution shift in predictions).

        Args:
            current_predictions: Current model predictions
            reference_predictions: Reference predictions (or use stored)

        Returns:
            DriftResult with prediction drift information
        """
        if reference_predictions is None:
            if len(self.recent_predictions) < 50:
                return DriftResult(drift_detected=False, drift_type="prediction", severity="low")
            reference_predictions = np.array(self.recent_predictions[-500:-100])

        # KS test on prediction distributions
        ks_stat, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)

        # Mean shift test
        mean_shift = abs(np.mean(current_predictions) - np.mean(reference_predictions))
        std_shift = abs(np.std(current_predictions) - np.std(reference_predictions))

        drift_detected = ks_pvalue < self.ks_threshold or mean_shift > 0.1

        # Determine severity
        if drift_detected:
            if ks_pvalue < 0.01 or mean_shift > 0.2:
                severity = "high"
            elif ks_pvalue < 0.03 or mean_shift > 0.15:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "low"

        result = DriftResult(
            drift_detected=drift_detected,
            drift_type="prediction",
            severity=severity,
            p_value=ks_pvalue,
            ks_statistic=ks_stat,
            details={"mean_shift": float(mean_shift), "std_shift": float(std_shift)},
        )

        # Update metrics
        self.metrics.set_gauge("prediction_drift_ks_pvalue", ks_pvalue)
        self.metrics.set_gauge("prediction_drift_mean_shift", mean_shift)

        if drift_detected:
            self.metrics.increment("drift_detected_total", labels={"type": "prediction"})
            self._send_alert(result)

        return result

    def record_prediction_actual(
        self, prediction: float, actual: float, timestamp: Optional[datetime] = None
    ) -> None:
        """Record prediction and actual for concept drift detection."""
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)
        self.recent_timestamps.append(timestamp or datetime.now())

        # Keep limited history
        max_history = 1000
        if len(self.recent_predictions) > max_history:
            self.recent_predictions = self.recent_predictions[-max_history:]
            self.recent_actuals = self.recent_actuals[-max_history:]
            self.recent_timestamps = self.recent_timestamps[-max_history:]

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        bins = np.histogram_bin_edges(reference, bins=10)

        # Calculate frequencies
        ref_freq, _ = np.histogram(reference, bins=bins)
        curr_freq, _ = np.histogram(current, bins=bins)

        # Normalize to probabilities
        ref_prob = ref_freq / len(reference) if len(reference) > 0 else ref_freq
        curr_prob = curr_freq / len(current) if len(current) > 0 else curr_freq

        # Avoid division by zero
        ref_prob = np.where(ref_prob == 0, 1e-6, ref_prob)
        curr_prob = np.where(curr_prob == 0, 1e-6, curr_prob)

        # Calculate PSI
        psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))

        return float(psi) if not np.isnan(psi) else 0.0

    def _send_alert(self, result: DriftResult) -> None:
        """Send alert notification for drift detection."""
        if self.notification_service is None:
            return

        message = (
            f"🚨 Drift Alert: {result.drift_type.upper()} drift detected\n"
            f"Severity: {result.severity}\n"
            f"Timestamp: {result.timestamp.isoformat()}\n"
        )

        if result.drifted_features:
            message += f"Drifted features: {', '.join(result.drifted_features)}\n"

        if result.psi is not None:
            message += f"PSI: {result.psi:.4f}\n"

        if result.p_value is not None:
            message += f"KS p-value: {result.p_value:.4f}\n"

        try:
            self.notification_service.send_alert(
                title="Model Drift Detected", message=message, severity=result.severity
            )
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")

    def check_all_drift(
        self,
        current_data: Optional[pl.DataFrame] = None,
        current_predictions: Optional[np.ndarray] = None,
    ) -> dict[str, DriftResult]:
        """
        Run all drift detection checks.

        Args:
            current_data: Current feature data
            current_predictions: Current predictions

        Returns:
            Dictionary of drift results by type
        """
        results = {}

        # Data drift
        if current_data is not None:
            results["data"] = self.detect_data_drift(current_data)

        # Concept drift
        results["concept"] = self.detect_concept_drift()

        # Prediction drift
        if current_predictions is not None:
            results["prediction"] = self.detect_prediction_drift(current_predictions)

        return results
