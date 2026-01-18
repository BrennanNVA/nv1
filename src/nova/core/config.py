"""Configuration management with TOML and Pydantic validation."""

import os
from pathlib import Path
from typing import Any, Optional

import pytomlpp
from pydantic import BaseModel, Field, field_validator, model_validator


class TechnicalConfig(BaseModel):
    """Technical indicator configuration."""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    atr_period: int = 14
    sma_short: int = 20
    sma_long: int = 50
    sma_trend: int = 200


class MLConfig(BaseModel):
    """Machine learning model configuration."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    optuna_trials: int = 100
    optuna_timeout: int = 3600
    # GPU optimization settings
    use_quantile_dmatrix: bool = True  # Use QuantileDMatrix for 5x memory reduction
    use_rmm: bool = True  # Use RAPIDS Memory Manager for faster allocation
    gradient_sampling: bool = (
        True  # Enable gradient-based sampling (XGBoost 3.0+) - Additional 20-30% memory savings
    )
    feature_selection_top_n: Optional[int] = None  # Select top N features (None = use all)
    feature_selection_prioritize_research: bool = (
        True  # Prioritize research-backed indicators (Squeeze_pro, PPO, MACD, ROC63, RSI63)
    )


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    risk_per_trade_pct: float = Field(default=0.02, ge=0.0, le=0.1)
    max_portfolio_risk_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    atr_stop_multiplier: float = 2.0
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = Field(default=0.05, ge=0.0, le=0.5)
    max_drawdown_pct: float = Field(default=0.10, ge=0.0, le=0.5)
    peak_equity_check_interval: int = 60


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    max_errors_per_minute: int = 5
    error_window_seconds: int = 60
    halt_on_breach: bool = True
    notification_on_breach: bool = True


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""

    model_name: str = "llama3"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = 512
    context_window: int = 4096
    bullish_keywords: list[str] = Field(
        default_factory=lambda: ["positive", "growth", "upside", "bullish", "strong"]
    )
    bearish_keywords: list[str] = Field(
        default_factory=lambda: ["negative", "decline", "downside", "bearish", "weak"]
    )
    neutral_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    """Data configuration."""

    symbols: list[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "AMD",
            "INTC",
            "NFLX",
            "SPY",
            "QQQ",
        ]
    )
    default_timeframe: str = "1Day"
    lookback_periods: int = 756  # 3 years minimum for reliable swing trading
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_db: str = "nova_aetus"
    timescale_user: str = "postgres"
    timescale_password: str = "postgres"
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://api.alpaca.markets"  # Pro API endpoint

    @field_validator("timescale_password", mode="before")
    @classmethod
    def load_password_from_env(cls, v: Any) -> str:
        """Load password from environment variable if available."""
        return os.getenv("TIMESCALE_PASSWORD", v if v else "postgres")

    @model_validator(mode="after")
    def load_alpaca_credentials_from_env(self) -> "DataConfig":
        """Load Alpaca credentials from environment variables after model creation."""
        # Prioritize environment variables over config file values
        env_api_key = os.getenv("ALPACA_API_KEY", "")
        env_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        if env_api_key:
            self.alpaca_api_key = env_api_key
        if env_secret_key:
            self.alpaca_secret_key = env_secret_key

        return self


class NotificationConfig(BaseModel):
    """Notification configuration."""

    discord_enabled: bool = True
    discord_webhook_url: str = ""
    notification_level: str = "INFO"

    @field_validator("discord_webhook_url", mode="before")
    @classmethod
    def load_webhook_from_env(cls, v: Any) -> str:
        """Load webhook URL from environment variable if available."""
        return os.getenv("DISCORD_WEBHOOK_URL", v if v else "")


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    port: int = 8501
    host: str = "localhost"
    theme: str = "dark"
    enable_metrics: bool = True


class ResourceConfig(BaseModel):
    """Resource management configuration."""

    max_db_connections: int = Field(default=10, ge=1, le=100)
    max_concurrent_requests: int = Field(default=50, ge=1, le=1000)
    max_memory_mb: Optional[int] = Field(default=None, ge=100)
    ollama_max_concurrent: int = Field(default=5, ge=1, le=20)
    thread_pool_size: int = Field(default=10, ge=1, le=50)


class Config(BaseModel):
    """Main configuration class."""

    technical: TechnicalConfig
    ml: MLConfig
    risk: RiskConfig
    circuit_breaker: CircuitBreakerConfig
    sentiment: SentimentConfig
    data: DataConfig
    notifications: NotificationConfig
    dashboard: DashboardConfig
    resources: Optional[ResourceConfig] = None


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to config.toml file. If None, searches in standard locations.

    Returns:
        Config object with validated settings.
    """
    if config_path is None:
        # Search in current directory and parent directories
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "config.toml",
            current_dir.parent / "config.toml",
            Path(__file__).parent.parent.parent.parent / "config.toml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find config.toml in any of: {possible_paths}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dict: dict[str, Any] = pytomlpp.load(config_path)
    return Config(**config_dict)
