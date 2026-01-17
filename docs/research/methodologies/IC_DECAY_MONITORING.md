# IC Decay Monitoring Methodology

## Overview

Information Coefficient (IC) decay monitoring is critical for detecting alpha decay and signal degradation in trading systems. This document describes the methodology implemented in Nova Aetus.

## What is IC?

Information Coefficient (IC) is the correlation between predicted returns and actual returns. It measures the predictive power of a signal.

- **IC > 0.05**: Strong signal
- **IC 0.02-0.05**: Moderate signal
- **IC < 0.02**: Weak signal

## IC Decay

Alpha decay is the natural degradation of signal quality over time. Research shows:
- **US markets**: ~5.6% annual decay
- **European markets**: ~9.9% annual decay

## Implementation

### Multi-Horizon IC Calculation

IC is calculated at multiple forward horizons:
- **1-day horizon**: Short-term signal quality
- **5-day horizon**: Medium-term signal quality
- **20-day horizon**: Long-term signal quality

### Signal Half-Life

Half-life is calculated using AR(1) / Ornstein-Uhlenbeck process:

```
Half-life = -log(2) / log(autocorrelation)
```

This estimates how long it takes for signal quality to decay by 50%.

### Rolling IC Monitoring

Rolling IC is calculated over a window (default: 20 periods) to track signal quality over time.

### Automated Alerts

Alerts are triggered when:
- IC drops below threshold (default: 0.05)
- IC decay rate exceeds threshold
- Half-life drops below threshold

## Usage

```python
from nova.monitoring import ICTracker

# Initialize tracker
tracker = ICTracker(
    horizons=[1, 5, 20],
    alert_threshold=0.05,
    rolling_window=20
)

# Register signal
tracker.register_signal("technical_signal")

# Record prediction
tracker.record_prediction("technical_signal", horizon=1, prediction=0.05)

# Record actual (later)
tracker.record_actual("technical_signal", horizon=1, actual=0.03)

# Get IC statistics
stats = tracker.get_ic_stats("technical_signal", horizon=1)
print(f"Mean IC: {stats.mean_ic:.4f}")
print(f"Half-life: {stats.half_life_days:.1f} days")
```

## Integration with Prometheus

IC metrics are automatically exported to Prometheus:
- `ic_{signal_name}_horizon_{horizon}`: Current IC value
- `ic_alert_total`: Total number of IC alerts

## References

- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
- "Alpha Decay in Financial Markets" (Academic research)
