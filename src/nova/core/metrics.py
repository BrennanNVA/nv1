"""Metrics collection for observability (Prometheus-style).

Research-backed implementation based on:
- Google SRE "Golden Signals" (Latency, Traffic, Errors, Saturation)
- Institutional-grade trading system observability
- Support for real-time alerting and historical analysis
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Institutional-grade metrics collector for system observability.

    Implements:
    - Counters for discrete events (trades, errors, heartbeats)
    - Gauges for point-in-time values (equity, drawdown, margin)
    - Histograms for distribution analysis (latency, slippage)
    - SRE Golden Signals tracking
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.start_time = datetime.now()

        # SRE Golden Signals shortcuts
        self._latency_signals = ["market_data_latency", "inference_latency", "execution_latency"]
        self._error_signals = ["api_errors", "model_errors", "execution_errors"]

    def increment(
        self, metric_name: str, value: float = 1.0, labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the metric
            value: Amount to increment
            labels: Optional labels
        """
        key = self._format_key(metric_name, labels)
        self.counters[key] += value

    def set_gauge(
        self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric.

        Args:
            metric_name: Name of the metric
            value: Gauge value
            labels: Optional labels
        """
        key = self._format_key(metric_name, labels)
        self.gauges[key] = value

    def record_histogram(
        self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record a histogram value.

        Args:
            metric_name: Name of the metric
            value: Value to record
            labels: Optional labels
        """
        key = self._format_key(metric_name, labels)
        self.histograms[key].append(value)
        # Keep only last 1000 values to prevent memory issues
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]

    def record_timing(
        self, metric_name: str, duration_seconds: float, labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record a timing metric (convenience for histogram).

        Args:
            metric_name: Name of the metric
            duration_seconds: Duration in seconds
            labels: Optional labels
        """
        self.record_histogram(metric_name, duration_seconds * 1000, labels)  # Convert to ms

    def record_latency(
        self, component: str, duration_seconds: float, labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record component latency (SRE Golden Signal).

        Args:
            component: Component name (e.g., 'inference', 'database')
            duration_seconds: Duration in seconds
            labels: Optional labels
        """
        self.record_timing(f"{component}_latency_ms", duration_seconds, labels)

    def record_error(
        self, component: str, error_type: str = "general", labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record a system error (SRE Golden Signal).

        Args:
            component: Component where error occurred
            error_type: Type of error
            labels: Optional labels
        """
        final_labels = labels or {}
        final_labels["component"] = component
        final_labels["error_type"] = error_type
        self.increment("system_errors_total", 1.0, final_labels)

    def _format_key(self, name: str, labels: Optional[dict[str, str]]) -> str:
        """Format metric key with labels for Prometheus."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all metrics with statistics.

        Returns:
            Dictionary with metrics data
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        metrics = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: self._calculate_histogram_stats(values)
                for name, values in self.histograms.items()
            },
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat(),
        }

        return metrics

    def _calculate_histogram_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {
                "count": 0,
                "sum": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
            }

        arr = np.array(values)
        return {
            "count": len(values),
            "sum": float(np.sum(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "avg": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_prometheus_format(self) -> str:
        """
        Get metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Counters
        for name, value in self.counters.items():
            base_name = name.split("{")[0]
            lines.append(f"# HELP {base_name} Total count of {base_name}")
            lines.append(f"# TYPE {base_name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in self.gauges.items():
            base_name = name.split("{")[0]
            lines.append(f"# HELP {base_name} Current value of {base_name}")
            lines.append(f"# TYPE {base_name} gauge")
            lines.append(f"{name} {value}")

        # Histograms
        for name, values in self.histograms.items():
            base_name = name.split("{")[0]
            stats = self._calculate_histogram_stats(values)

            lines.append(f"# HELP {base_name} Statistics for {base_name}")
            lines.append(f"# TYPE {base_name} summary")

            labels = ""
            if "{" in name:
                labels = name[name.find("{") + 1 : name.find("}")]
                if labels:
                    labels += ","

            lines.append(f'{base_name}{{{labels}quantile="0.5"}} {stats["p50"]}')
            lines.append(f'{base_name}{{{labels}quantile="0.95"}} {stats["p95"]}')
            lines.append(f'{base_name}{{{labels}quantile="0.99"}} {stats["p99"]}')
            lines.append(f'{base_name}_sum{{{labels}}} {stats["sum"]}')
            lines.append(f'{base_name}_count{{{labels}}} {stats["count"]}')

        return "\n".join(lines)


class MetricsTimer:
    """Context manager for timing operations."""

    def __init__(self, metrics: MetricsCollector, metric_name: str, labels: Optional[dict] = None):
        """
        Initialize timer.

        Args:
            metrics: MetricsCollector instance
            metric_name: Name of the metric
            labels: Optional labels
        """
        self.metrics = metrics
        self.metric_name = metric_name
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_timing(self.metric_name, duration, self.labels)


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
