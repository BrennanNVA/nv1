"""Async data fetching for market data and fundamentals."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import polars as pl

from ..core.config import DataConfig
from ..core.validation import SymbolValidator, TimestampValidator

logger = logging.getLogger(__name__)

# Lazy imports to avoid dependency issues
try:
    from alpaca.common.exceptions import APIError
    from alpaca.data import StockBarsRequest, StockQuotesRequest, StockTradesRequest, TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not available, Alpaca data fetching disabled")

try:
    from yahooquery import Ticker

    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    logger.warning("yahooquery not available, Yahoo fallback disabled")


class DataLoader:
    """Async data loader for fetching market data."""

    def __init__(self, config: DataConfig) -> None:
        """
        Initialize data loader.

        Args:
            config: Data configuration
        """
        self.config = config
        self.alpaca_client: Optional[StockHistoricalDataClient] = None

        # Initialize Alpaca client if credentials available
        if ALPACA_AVAILABLE and config.alpaca_api_key and config.alpaca_secret_key:
            try:
                # Pro API uses https://api.alpaca.markets (default when sandbox=False)
                # For custom endpoints, use url_override parameter
                client_params = {
                    "api_key": config.alpaca_api_key,
                    "secret_key": config.alpaca_secret_key,
                    "sandbox": False,  # Pro API is not sandbox
                }
                # Only use url_override if base_url is different from default
                if (
                    config.alpaca_base_url
                    and config.alpaca_base_url != "https://api.alpaca.markets"
                ):
                    client_params["url_override"] = config.alpaca_base_url

                self.alpaca_client = StockHistoricalDataClient(**client_params)
                logger.info("Alpaca client initialized (Pro API)")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca client: {e}")

        logger.info(f"DataLoader initialized with {len(config.symbols)} symbols")

    async def fetch_historical_bars(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[str] = None,
        extended_hours: bool = False,
        adjustment: str = "raw",
    ) -> pl.DataFrame:
        # Validate inputs
        symbol = SymbolValidator.validate(symbol)
        if start_date:
            start_date = TimestampValidator.validate_datetime(start_date)
        if end_date:
            end_date = TimestampValidator.validate_datetime(end_date)
        """
        Fetch historical bar data for a symbol.

        Tries Alpaca first, falls back to yahooquery if Alpaca fails.

        Args:
            symbol: Stock symbol
            start_date: Start date (defaults to lookback_periods ago)
            end_date: End date (defaults to now)
            timeframe: Timeframe (defaults to config.default_timeframe)
            extended_hours: Include pre/post market data (Pro feature)
            adjustment: Price adjustment - "raw", "split", "dividend", or "all"

        Returns:
            DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            # Calculate start date based on lookback periods
            if timeframe is None:
                timeframe = self.config.default_timeframe

            # Estimate days based on timeframe
            days_per_period = self._get_days_per_period(timeframe)
            start_date = end_date - timedelta(days=self.config.lookback_periods * days_per_period)

        if timeframe is None:
            timeframe = self.config.default_timeframe

        logger.debug(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

        # Try Alpaca first
        if self.alpaca_client:
            try:
                df = await self._fetch_from_alpaca(
                    symbol, start_date, end_date, timeframe, extended_hours, adjustment
                )
                if not df.is_empty():
                    logger.info(
                        f"Successfully fetched {len(df)} bars for {symbol} from Alpaca (Pro API)"
                    )
                    return df
            except Exception as e:
                logger.warning(f"Alpaca fetch failed for {symbol}: {e}, trying yahooquery fallback")

        # Fallback to yahooquery
        if YAHOOQUERY_AVAILABLE:
            try:
                df = await self._fetch_from_yahooquery(symbol, start_date, end_date, timeframe)
                if not df.is_empty():
                    logger.info(f"Successfully fetched {len(df)} bars for {symbol} from yahooquery")
                    return df
            except Exception as e:
                logger.error(f"yahooquery fetch failed for {symbol}: {e}")
        else:
            logger.error("No data sources available (Alpaca failed, yahooquery not installed)")

        # Return empty DataFrame if all sources fail
        logger.warning(f"No data fetched for {symbol}, returning empty DataFrame")
        schema = {
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
        return pl.DataFrame(schema=schema)

    async def _fetch_from_alpaca(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        extended_hours: bool = False,
        adjustment: str = "raw",
    ) -> pl.DataFrame:
        """
        Fetch data from Alpaca Pro API.

        Pro Trader subscription benefits:
        - Full SIP feed (all US exchanges, not just IEX)
        - Extended hours data (pre/post market)
        - No 15-minute delay on recent data
        - Higher rate limits (10,000 calls/min)
        - Better precision and lower latency

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string (e.g., "1Day", "4Hour")
            extended_hours: Include pre/post market data (Pro feature)
            adjustment: Price adjustment - "raw", "split", "dividend", or "all"

        Returns:
            DataFrame with OHLCV data
        """
        if not ALPACA_AVAILABLE or not self.alpaca_client:
            raise RuntimeError("Alpaca client not available")

        # Convert timeframe string to Alpaca TimeFrame
        alpaca_timeframe = self._convert_timeframe(timeframe)

        # Run Alpaca API call in executor (Alpaca SDK is sync)
        loop = asyncio.get_event_loop()

        def _fetch():
            # Map adjustment string to Alpaca adjustment enum
            adj_map = {
                "raw": Adjustment.RAW,
                "split": Adjustment.SPLIT,
                "dividend": Adjustment.DIVIDEND,
                "all": Adjustment.ALL,
            }
            adj_enum = adj_map.get(adjustment.lower(), Adjustment.RAW)

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date,
                adjustment=adj_enum,
                feed=DataFeed.SIP,  # Pro: Full SIP feed (all exchanges)
            )
            bars = self.alpaca_client.get_stock_bars(request_params)
            return bars.data.get(symbol, [])

        bars_list = await loop.run_in_executor(None, _fetch)

        if not bars_list:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        # Convert to Polars DataFrame
        data = {
            "timestamp": [bar.timestamp for bar in bars_list],
            "open": [float(bar.open) for bar in bars_list],
            "high": [float(bar.high) for bar in bars_list],
            "low": [float(bar.low) for bar in bars_list],
            "close": [float(bar.close) for bar in bars_list],
            "volume": [int(bar.volume) for bar in bars_list],
        }

        df = pl.DataFrame(data)
        df = df.sort("timestamp")  # Ensure chronological order
        return df

    async def _fetch_from_yahooquery(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pl.DataFrame:
        """
        Fetch data from yahooquery as fallback.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe string (e.g., "1Day", "4Hour")

        Returns:
            DataFrame with OHLCV data
        """
        if not YAHOOQUERY_AVAILABLE:
            raise RuntimeError("yahooquery not available")

        # Run yahooquery API call in executor (yahooquery is sync)
        loop = asyncio.get_event_loop()

        # Import pandas here for use in _fetch
        try:
            import pandas as pd  # noqa: I001
        except ImportError:
            pd = None

        def _fetch():
            ticker = Ticker(symbol)
            # yahooquery uses different period format
            # Convert datetime to yahooquery compatible format
            history = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=self._convert_timeframe_for_yahooquery(timeframe),
            )

            if history is None or history.empty:
                return None

            # Reset index to get date as column (yahooquery uses MultiIndex with symbol and date)
            if pd is not None and isinstance(history.index, pd.MultiIndex):
                history = history.reset_index()
                # Remove symbol column if present (we only fetch one symbol at a time)
                if "symbol" in history.columns:
                    history = history.drop(columns=["symbol"])
                if "date" in history.columns:
                    history = history.rename(columns={"date": "timestamp"})
            elif (
                pd is not None and hasattr(history.index, "names") and "date" in history.index.names
            ):
                history = history.reset_index()
                if "date" in history.columns:
                    history = history.rename(columns={"date": "timestamp"})

            return history

        history_df = await loop.run_in_executor(None, _fetch)

        if history_df is None or history_df.empty:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        # Convert pandas DataFrame to Polars
        # yahooquery returns pandas DataFrame
        if pd is not None and isinstance(history_df, pd.DataFrame):
            df = pl.from_pandas(history_df)
        else:
            df = history_df

        # Ensure column names match expected schema
        column_mapping = {
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        # Select and rename columns
        available_cols = df.columns
        selected_cols = []
        for target_col, source_col in column_mapping.items():
            if source_col in available_cols:
                selected_cols.append(source_col)
            elif source_col.upper() in available_cols:
                selected_cols.append(source_col.upper())

        df = df.select(selected_cols)
        df = df.rename({col: col.lower() for col in df.columns})

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort by timestamp
        df = df.sort("timestamp")

        # Ensure all required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                if col == "volume":
                    df = df.with_columns(pl.lit(0).alias("volume").cast(pl.Int64))
                else:
                    logger.warning(f"Missing column {col} in yahooquery data for {symbol}")

        return df.select(required_cols)

    def _convert_timeframe(self, timeframe: str) -> TimeFrame:
        """
        Convert timeframe string to Alpaca TimeFrame.

        Args:
            timeframe: Timeframe string (e.g., "1Day", "4Hour", "15Min")

        Returns:
            Alpaca TimeFrame object
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError("Alpaca not available")

        timeframe_lower = timeframe.lower().strip()

        # Parse number and unit
        if "day" in timeframe_lower or "d" in timeframe_lower:
            if "1" in timeframe_lower or timeframe_lower.startswith("d"):
                return TimeFrame.Day
            else:
                # Multi-day
                num = int("".join(filter(str.isdigit, timeframe_lower)) or "1")
                return TimeFrame(amount=num, unit=TimeFrame.Unit.Day)
        elif "hour" in timeframe_lower or "h" in timeframe_lower:
            num = int("".join(filter(str.isdigit, timeframe_lower)) or "1")
            return TimeFrame(amount=num, unit=TimeFrame.Unit.Hour)
        elif "min" in timeframe_lower or "minute" in timeframe_lower:
            num = int("".join(filter(str.isdigit, timeframe_lower)) or "1")
            return TimeFrame(amount=num, unit=TimeFrame.Unit.Minute)
        else:
            # Default to daily
            return TimeFrame.Day

    def _convert_timeframe_for_yahooquery(self, timeframe: str) -> str:
        """
        Convert timeframe string to yahooquery interval format.

        Args:
            timeframe: Timeframe string (e.g., "1Day", "4Hour", "15Min")

        Returns:
            yahooquery interval string (e.g., "1d", "4h", "15m")
        """
        timeframe_lower = timeframe.lower().strip()

        if "day" in timeframe_lower or "d" in timeframe_lower:
            num = "".join(filter(str.isdigit, timeframe_lower)) or "1"
            return f"{num}d"
        elif "hour" in timeframe_lower or "h" in timeframe_lower:
            num = "".join(filter(str.isdigit, timeframe_lower)) or "1"
            return f"{num}h"
        elif "min" in timeframe_lower or "minute" in timeframe_lower:
            num = "".join(filter(str.isdigit, timeframe_lower)) or "1"
            return f"{num}m"
        else:
            return "1d"  # Default to daily

    async def fetch_multiple_symbols(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch historical data for multiple symbols concurrently.

        Args:
            symbols: List of symbols (defaults to config.symbols)
            start_date: Start date
            end_date: End date
            timeframe: Timeframe

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.config.symbols

        logger.info(f"Fetching data for {len(symbols)} symbols concurrently")

        tasks = [
            self.fetch_historical_bars(symbol, start_date, end_date, timeframe)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)

        return dict(zip(symbols, results, strict=False))

    async def fetch_fundamentals(self, symbol: str) -> dict:
        """
        Fetch fundamental data for a symbol using yahooquery.

        Fetches key ratios (P/E, P/B, ROE) and calculates simple value/quality factors.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental metrics:
            {
                'pe_ratio': float,
                'pb_ratio': float,
                'roe': float,
                'debt_to_equity': float,
                'current_ratio': float,
                'value_score': float,  # -1 to 1 (lower P/E/P/B = higher score)
                'quality_score': float,  # -1 to 1 (higher ROE, lower debt = higher score)
            }
        """
        logger.debug(f"Fetching fundamentals for {symbol}")

        if not YAHOOQUERY_AVAILABLE:
            logger.warning("yahooquery not available, fundamental data fetching disabled")
            return {}

        try:
            # Run yahooquery API call in executor (yahooquery is sync)
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: Ticker(symbol))

            # Fetch key statistics and financial data
            key_stats = await loop.run_in_executor(None, lambda: ticker.key_stats)
            summary_detail = await loop.run_in_executor(None, lambda: ticker.summary_detail)
            financial_data = await loop.run_in_executor(None, lambda: ticker.financial_data)

            # Extract metrics
            fundamentals = {}

            # P/E Ratio
            if symbol in summary_detail and summary_detail[symbol].get("trailingPE"):
                fundamentals["pe_ratio"] = float(summary_detail[symbol]["trailingPE"])
            elif symbol in key_stats and key_stats[symbol].get("trailingPE"):
                fundamentals["pe_ratio"] = float(key_stats[symbol]["trailingPE"])
            else:
                fundamentals["pe_ratio"] = None

            # P/B Ratio
            if symbol in key_stats and key_stats[symbol].get("priceToBook"):
                fundamentals["pb_ratio"] = float(key_stats[symbol]["priceToBook"])
            else:
                fundamentals["pb_ratio"] = None

            # ROE (Return on Equity)
            if symbol in key_stats and key_stats[symbol].get("returnOnEquity"):
                roe_raw = key_stats[symbol]["returnOnEquity"]
                # ROE is often returned as a percentage (e.g., 15.5 means 15.5%)
                fundamentals["roe"] = (
                    float(roe_raw) / 100
                    if roe_raw and roe_raw > 1
                    else float(roe_raw)
                    if roe_raw
                    else None
                )
            else:
                fundamentals["roe"] = None

            # Debt to Equity
            if symbol in key_stats and key_stats[symbol].get("debtToEquity"):
                fundamentals["debt_to_equity"] = float(key_stats[symbol]["debtToEquity"])
            else:
                fundamentals["debt_to_equity"] = None

            # Current Ratio
            if symbol in key_stats and key_stats[symbol].get("currentRatio"):
                fundamentals["current_ratio"] = float(key_stats[symbol]["currentRatio"])
            else:
                fundamentals["current_ratio"] = None

            # Calculate value score (lower P/E and P/B = better value)
            value_score = 0.0
            if fundamentals["pe_ratio"] and fundamentals["pb_ratio"]:
                # Normalize P/E: lower is better, assume reasonable range 5-50
                pe_score = (
                    max(0, min(1, (50 - fundamentals["pe_ratio"]) / 45))
                    if fundamentals["pe_ratio"] > 0
                    else 0.5
                )
                # Normalize P/B: lower is better, assume reasonable range 0.5-5
                pb_score = (
                    max(0, min(1, (5 - fundamentals["pb_ratio"]) / 4.5))
                    if fundamentals["pb_ratio"] > 0
                    else 0.5
                )
                value_score = (pe_score + pb_score) / 2
                # Convert to -1 to 1 range
                value_score = (value_score - 0.5) * 2
            elif fundamentals["pe_ratio"]:
                pe_score = (
                    max(0, min(1, (50 - fundamentals["pe_ratio"]) / 45))
                    if fundamentals["pe_ratio"] > 0
                    else 0.5
                )
                value_score = (pe_score - 0.5) * 2
            elif fundamentals["pb_ratio"]:
                pb_score = (
                    max(0, min(1, (5 - fundamentals["pb_ratio"]) / 4.5))
                    if fundamentals["pb_ratio"] > 0
                    else 0.5
                )
                value_score = (pb_score - 0.5) * 2

            fundamentals["value_score"] = value_score

            # Calculate quality score (higher ROE, lower debt = better quality)
            quality_score = 0.0
            if fundamentals["roe"] is not None:
                # ROE: higher is better, assume reasonable range -0.1 to 0.3 (10% to 30%)
                roe_score = max(0, min(1, (fundamentals["roe"] + 0.1) / 0.4))
                quality_score = roe_score

            if fundamentals["debt_to_equity"] is not None:
                # Debt/Equity: lower is better, assume reasonable range 0-2
                debt_score = max(0, min(1, (2 - fundamentals["debt_to_equity"]) / 2))
                if quality_score > 0:
                    quality_score = (quality_score + debt_score) / 2
                else:
                    quality_score = debt_score

            # Convert to -1 to 1 range
            if quality_score > 0:
                quality_score = (quality_score - 0.5) * 2

            fundamentals["quality_score"] = quality_score

            logger.info(
                f"Fundamentals fetched for {symbol}: "
                f"P/E={fundamentals.get('pe_ratio')}, P/B={fundamentals.get('pb_ratio')}, "
                f"ROE={fundamentals.get('roe')}, Value={fundamentals.get('value_score'):.2f}, "
                f"Quality={fundamentals.get('quality_score'):.2f}"
            )

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}", exc_info=True)
            return {}

    async def fetch_fundamentals_batch(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch fundamental data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to fundamental metrics
        """
        logger.info(f"Fetching fundamentals for {len(symbols)} symbols")

        tasks = [self.fetch_fundamentals(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fundamentals_dict = {}
        for symbol, result in zip(symbols, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Error fetching fundamentals for {symbol}: {result}")
                fundamentals_dict[symbol] = {}
            else:
                fundamentals_dict[symbol] = result

        return fundamentals_dict

    async def fetch_trades(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Fetch trade-level data from Alpaca Pro API.

        Pro feature: Access to individual trades (not just aggregated bars).
        Useful for microstructure analysis, volume profile, and order flow.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with trade data (timestamp, price, size, exchange)
        """
        if not ALPACA_AVAILABLE or not self.alpaca_client:
            raise RuntimeError("Alpaca client not available")

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1)  # Default to last 24 hours

        loop = asyncio.get_event_loop()

        def _fetch():
            request = StockTradesRequest(
                symbol_or_symbols=[symbol],
                start=start_date,
                end=end_date,
                feed=DataFeed.SIP,  # Pro: Full SIP feed (all exchanges)
            )
            trades = self.alpaca_client.get_stock_trades(request)
            return trades.data.get(symbol, [])

        trades_list = await loop.run_in_executor(None, _fetch)

        if not trades_list:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "price": pl.Float64,
                    "size": pl.Int64,
                    "exchange": pl.String,
                }
            )

        data = {
            "timestamp": [trade.timestamp for trade in trades_list],
            "price": [float(trade.price) for trade in trades_list],
            "size": [int(trade.size) for trade in trades_list],
            "exchange": [str(trade.exchange) for trade in trades_list],
        }

        df = pl.DataFrame(data).sort("timestamp")
        logger.info(f"Fetched {len(df)} trades for {symbol} from Alpaca Pro")
        return df

    async def fetch_quotes(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Fetch quote-level data from Alpaca Pro API.

        Pro feature: Access to bid/ask quotes with full depth.
        Useful for spread analysis, market depth, and liquidity assessment.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with quote data (timestamp, bid_price, ask_price, bid_size, ask_size)
        """
        if not ALPACA_AVAILABLE or not self.alpaca_client:
            raise RuntimeError("Alpaca client not available")

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1)  # Default to last 24 hours

        loop = asyncio.get_event_loop()

        def _fetch():
            request = StockQuotesRequest(
                symbol_or_symbols=[symbol],
                start=start_date,
                end=end_date,
                feed=DataFeed.SIP,  # Pro: Full SIP feed (all exchanges)
            )
            quotes = self.alpaca_client.get_stock_quotes(request)
            return quotes.data.get(symbol, [])

        quotes_list = await loop.run_in_executor(None, _fetch)

        if not quotes_list:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "bid_price": pl.Float64,
                    "ask_price": pl.Float64,
                    "bid_size": pl.Int64,
                    "ask_size": pl.Int64,
                }
            )

        data = {
            "timestamp": [quote.timestamp for quote in quotes_list],
            "bid_price": [float(quote.bid_price) for quote in quotes_list],
            "ask_price": [float(quote.ask_price) for quote in quotes_list],
            "bid_size": [int(quote.bid_size) for quote in quotes_list],
            "ask_size": [int(quote.ask_size) for quote in quotes_list],
        }

        df = pl.DataFrame(data).sort("timestamp")
        logger.info(f"Fetched {len(df)} quotes for {symbol} from Alpaca Pro")
        return df

    def _get_days_per_period(self, timeframe: str) -> int:
        """
        Estimate days per period for a given timeframe.

        Args:
            timeframe: Timeframe string (e.g., "1Day", "4Hour")

        Returns:
            Estimated days per period
        """
        timeframe_lower = timeframe.lower()

        if "day" in timeframe_lower or "d" in timeframe_lower:
            return 1
        elif "hour" in timeframe_lower or "h" in timeframe_lower:
            return 1 / 24
        elif "week" in timeframe_lower or "w" in timeframe_lower:
            return 7
        elif "month" in timeframe_lower or "m" in timeframe_lower:
            return 30
        else:
            # Default to daily
            return 1
