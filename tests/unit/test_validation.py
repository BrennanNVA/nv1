"""Unit tests for validation module."""

from datetime import datetime

import polars as pl
import pytest

from src.nova.core.validation import (
    DataFrameValidator,
    SQLQueryValidator,
    SymbolValidator,
    TimestampValidator,
)


class TestSymbolValidator:
    """Test symbol validation."""

    def test_valid_symbols(self):
        """Test valid symbol formats."""
        assert SymbolValidator.validate("AAPL") == "AAPL"
        assert SymbolValidator.validate("msft") == "MSFT"  # Normalized to uppercase
        assert SymbolValidator.validate("GOOGL") == "GOOGL"

    def test_invalid_symbols(self):
        """Test invalid symbol formats."""
        with pytest.raises(ValueError):
            SymbolValidator.validate("")

        with pytest.raises(ValueError):
            SymbolValidator.validate("AAPL123")  # Contains numbers

        with pytest.raises(ValueError):
            SymbolValidator.validate("aa-pl")  # Contains invalid characters

    def test_symbol_list(self):
        """Test validating list of symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        validated = SymbolValidator.validate_list(symbols)
        assert validated == symbols


class TestTimestampValidator:
    """Test timestamp validation."""

    def test_datetime_object(self):
        """Test datetime object validation."""
        dt = datetime.now()
        result = TimestampValidator.validate_datetime(dt)
        assert result == dt

    def test_string_format(self):
        """Test string datetime validation."""
        result = TimestampValidator.validate_datetime("2024-01-15")
        assert isinstance(result, datetime)

    def test_invalid_format(self):
        """Test invalid datetime format."""
        with pytest.raises(ValueError):
            TimestampValidator.validate_datetime("invalid-date")


class TestDataFrameValidator:
    """Test DataFrame validation."""

    def test_valid_ohlcv(self, sample_ohlcv_data):
        """Test valid OHLCV DataFrame."""
        result = DataFrameValidator.validate_ohlcv(sample_ohlcv_data)
        assert result is not None

    def test_missing_columns(self):
        """Test DataFrame with missing columns."""
        df = pl.DataFrame({"close": [100, 101]})
        with pytest.raises(ValueError):
            DataFrameValidator.validate_ohlcv(df)

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
        with pytest.raises(ValueError):
            DataFrameValidator.validate_ohlcv(df)


class TestSQLQueryValidator:
    """Test SQL query validation."""

    def test_valid_select(self):
        """Test valid SELECT query."""
        query = "SELECT * FROM market_bars WHERE symbol = 'AAPL'"
        result = SQLQueryValidator.validate_select_only(query)
        assert result == query.strip()

    def test_dangerous_keywords(self):
        """Test queries with dangerous keywords."""
        with pytest.raises(ValueError):
            SQLQueryValidator.validate_select_only("DROP TABLE market_bars")

        with pytest.raises(ValueError):
            SQLQueryValidator.validate_select_only("DELETE FROM market_bars")

    def test_table_name_validation(self):
        """Test table name validation."""
        assert SQLQueryValidator.validate_table_name("market_bars") == "market_bars"

        with pytest.raises(ValueError):
            SQLQueryValidator.validate_table_name("market_bars; DROP TABLE")
