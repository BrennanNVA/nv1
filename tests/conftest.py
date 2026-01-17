"""Pytest configuration and fixtures."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame for testing."""
    from datetime import datetime, timedelta

    import polars as pl

    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i for i in range(10)],
            "high": [102.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "volume": [1000000] * 10,
        }
    )


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    from src.nova.core.config import (
        CircuitBreakerConfig,
        Config,
        DashboardConfig,
        DataConfig,
        MLConfig,
        NotificationConfig,
        RiskConfig,
        SentimentConfig,
        TechnicalConfig,
    )

    return Config(
        technical=TechnicalConfig(),
        ml=MLConfig(),
        risk=RiskConfig(),
        circuit_breaker=CircuitBreakerConfig(),
        sentiment=SentimentConfig(),
        data=DataConfig(symbols=["AAPL", "MSFT"]),
        notifications=NotificationConfig(discord_enabled=False),
        dashboard=DashboardConfig(),
    )
