"""Input validation at system boundaries."""

import logging
import re
from datetime import datetime
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class SymbolValidator:
    """Validate stock symbols."""

    # Valid symbol pattern: 1-5 uppercase letters, optionally followed by exchange
    SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]+)?$")
    MAX_SYMBOL_LENGTH = 10

    @classmethod
    def validate(cls, symbol: str) -> str:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Normalized symbol (uppercase)

        Raises:
            ValueError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ValueError(f"Symbol must be a string, got {type(symbol)}")

        symbol = symbol.strip().upper()

        if not symbol:
            raise ValueError("Symbol cannot be empty")

        if len(symbol) > cls.MAX_SYMBOL_LENGTH:
            raise ValueError(f"Symbol too long (max {cls.MAX_SYMBOL_LENGTH} chars): {symbol}")

        if not cls.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")

        return symbol

    @classmethod
    def validate_list(cls, symbols: list[str]) -> list[str]:
        """Validate a list of symbols."""
        return [cls.validate(s) for s in symbols]


class TimestampValidator:
    """Validate timestamps."""

    @classmethod
    def validate_datetime(cls, dt: Any) -> datetime:
        """
        Validate and normalize datetime.

        Args:
            dt: Datetime object, string, or timestamp

        Returns:
            Normalized datetime object

        Raises:
            ValueError: If datetime is invalid
        """
        if isinstance(dt, datetime):
            return dt

        if isinstance(dt, str):
            # Try common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(dt, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Invalid datetime format: {dt}")

        if isinstance(dt, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(dt)

        raise ValueError(f"Invalid datetime type: {type(dt)}")


class DataFrameValidator:
    """Validate Polars DataFrames."""

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    @classmethod
    def validate_ohlcv(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        Validate OHLCV DataFrame structure and data.

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If DataFrame is invalid
        """
        if df.is_empty():
            raise ValueError("DataFrame is empty")

        # Check required columns
        missing_cols = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data types
        if df["timestamp"].dtype != pl.Datetime:
            raise ValueError("timestamp column must be Datetime type")

        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if df[col].dtype not in (pl.Float64, pl.Float32):
                raise ValueError(f"{col} column must be numeric")
            # Check for negative values
            if (df[col] < 0).any():
                raise ValueError(f"{col} contains negative values")

        if df["volume"].dtype not in (pl.Int64, pl.Int32, pl.Float64):
            raise ValueError("volume column must be numeric")
        if (df["volume"] < 0).any():
            raise ValueError("volume contains negative values")

        # Validate OHLC relationships
        invalid_high = (df["high"] < df["low"]).any()
        if invalid_high:
            raise ValueError("high < low found in data")

        invalid_open = ((df["open"] < df["low"]) | (df["open"] > df["high"])).any()
        if invalid_open:
            logger.warning("Some open prices are outside high/low range")

        invalid_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).any()
        if invalid_close:
            logger.warning("Some close prices are outside high/low range")

        # Check for duplicates
        if "symbol" in df.columns and df.select("symbol", "timestamp").is_duplicated().any():
            logger.warning("Duplicate symbol/timestamp pairs found")

        return df


class SQLQueryValidator:
    """Validate SQL queries for safety."""

    DANGEROUS_KEYWORDS = {
        "DROP",
        "DELETE",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "INSERT",
        "UPDATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
    }

    ALLOWED_KEYWORDS = {"SELECT", "WITH", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "LIMIT"}

    @classmethod
    def validate_select_only(cls, query: str) -> str:
        """
        Validate that query only contains SELECT operations.

        Args:
            query: SQL query string

        Returns:
            Normalized query

        Raises:
            ValueError: If query contains dangerous operations
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        query_upper = query.strip().upper()

        if not query_upper:
            raise ValueError("Query cannot be empty")

        # Must start with SELECT or WITH (CTE)
        if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
            raise ValueError("Only SELECT queries are allowed")

        # Check for dangerous keywords
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                raise ValueError(f"Dangerous keyword '{keyword}' not allowed in queries")

        return query.strip()

    @classmethod
    def validate_table_name(cls, table_name: str) -> str:
        """
        Validate table name to prevent SQL injection.

        Args:
            table_name: Table name to validate

        Returns:
            Validated table name

        Raises:
            ValueError: If table name is invalid
        """
        if not isinstance(table_name, str):
            raise ValueError("Table name must be a string")

        table_name = table_name.strip()

        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name format: {table_name}")

        return table_name


class ConfigurationValidator:
    """Validate configuration values."""

    @classmethod
    def validate_percentage(
        cls, value: float, name: str, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        """
        Validate percentage value.

        Args:
            value: Percentage value (0.0 to 1.0)
            name: Name of the parameter (for error messages)
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated value

        Raises:
            ValueError: If value is out of range
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")

        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

        return float(value)

    @classmethod
    def validate_positive_int(cls, value: int, name: str, min_val: int = 1) -> int:
        """Validate positive integer."""
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")

        if value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        return value


def validate_webhook_url(url: str) -> str:
    """
    Validate Discord webhook URL.

    Args:
        url: Webhook URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("Webhook URL must be a string")

    url = url.strip()

    if not url:
        raise ValueError("Webhook URL cannot be empty")

    if not url.startswith("https://discord.com/api/webhooks/"):
        raise ValueError("Invalid Discord webhook URL format")

    # Basic format check: should have webhook ID and token
    parts = url.split("/")
    if len(parts) < 6:
        raise ValueError("Invalid Discord webhook URL format")

    return url
