"""Polars-based technical indicators with fractional differencing.

Research-backed implementation based on:
- "Key technical indicators for stock market prediction" (MLWA, 2025)
- "Advances in Financial Machine Learning" (Lopez de Prado) - Fractional differentiation
- High-value indicators: RSI, PPO, MACD, Bollinger Bands, ATR, OBV, VWAP
"""

import logging
from typing import Optional

import numpy as np
import polars as pl

from ..core.config import TechnicalConfig

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Technical indicator calculations using Polars with fractional differencing.

    Implements institutional-grade feature engineering:
    - Momentum: RSI, PPO, ROC, Squeeze Pro
    - Trend: MACD, Ichimoku, SAR, EMA crossovers
    - Volatility: ATR, Bollinger Bands, Keltner Channels
    - Volume: OBV, VWAP deviation, Volume ratio
    - Fractional differencing for stationarity
    - Z-score normalization
    """

    def __init__(self, config: TechnicalConfig) -> None:
        """
        Initialize technical features calculator.

        Args:
            config: Technical indicator configuration
        """
        self.config = config
        logger.info("TechnicalFeatures initialized with fractional differencing support")

    def calculate_sma(self, df: pl.DataFrame, period: Optional[int] = None) -> pl.Series:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with 'close' column
            period: Period for SMA (defaults to config.sma_short)

        Returns:
            Series with SMA values
        """
        if period is None:
            period = self.config.sma_short

        return df["close"].rolling_mean(window_size=period)

    def calculate_ema(
        self, df: pl.DataFrame, period: Optional[int] = None, column: str = "close"
    ) -> pl.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price column
            period: Period for EMA
            column: Column name to calculate EMA on (default: "close")

        Returns:
            Series with EMA values
        """
        if period is None:
            period = self.config.sma_short

        # Polars exponential moving average
        alpha = 2.0 / (period + 1.0)
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in DataFrame. Available columns: {df.columns}"
            )
        return df[column].ewm_mean(alpha=alpha, adjust=False)

    def calculate_rsi(self, df: pl.DataFrame, period: Optional[int] = None) -> pl.Series:
        """
        Calculate Relative Strength Index.

        Args:
            df: DataFrame with 'close' column
            period: Period for RSI (defaults to config.rsi_period)

        Returns:
            Series with RSI values (0-100)
        """
        if period is None:
            period = self.config.rsi_period

        close = df["close"]
        delta = close.diff().fill_null(0.0)

        # Separate gains and losses (keep same length as delta)
        gains = pl.when(delta > 0).then(delta).otherwise(0.0)
        losses = pl.when(delta < 0).then(-delta).otherwise(0.0)

        # Create temporary DataFrame by evaluating expressions
        temp_df = df.with_columns(
            [
                gains.alias("gains"),
                losses.alias("losses"),
            ]
        )

        # Calculate average gain and loss using Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / period
        avg_gain = temp_df["gains"].ewm_mean(alpha=alpha, adjust=False)
        avg_loss = temp_df["losses"].ewm_mean(alpha=alpha, adjust=False)

        # Calculate RS and RSI (handle division by zero)
        rs = avg_gain / avg_loss.replace({0.0: None})
        rsi = 100.0 - (100.0 / (1.0 + rs.fill_null(1.0)))

        return rsi.fill_null(50.0)  # Default to 50 if no data

    def calculate_macd(
        self,
        df: pl.DataFrame,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> dict[str, pl.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (defaults to config.macd_fast)
            slow: Slow EMA period (defaults to config.macd_slow)
            signal: Signal line period (defaults to config.macd_signal)

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        if fast is None:
            fast = self.config.macd_fast
        if slow is None:
            slow = self.config.macd_slow
        if signal is None:
            signal = self.config.macd_signal

        # Calculate EMAs
        ema_fast = self.calculate_ema(df, period=fast)
        ema_slow = self.calculate_ema(df, period=slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD line)
        # Create temporary DataFrame for signal calculation
        # Convert Series to list to avoid Expr issues
        macd_df = pl.DataFrame({"macd": macd_line.to_list()})
        signal_line = self.calculate_ema(macd_df, period=signal, column="macd")

        # Histogram
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    def calculate_bollinger_bands(
        self,
        df: pl.DataFrame,
        period: Optional[int] = None,
        stddev: Optional[float] = None,
    ) -> dict[str, pl.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'close' column
            period: Period for SMA (defaults to config.bollinger_period)
            stddev: Standard deviation multiplier (defaults to config.bollinger_stddev)

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' series
        """
        if period is None:
            period = self.config.bollinger_period
        if stddev is None:
            stddev = self.config.bollinger_stddev

        # Middle band (SMA)
        middle = self.calculate_sma(df, period=period)

        # Standard deviation
        std = df["close"].rolling_std(window_size=period)

        # Upper and lower bands
        upper = middle + (std * stddev)
        lower = middle - (std * stddev)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    def calculate_atr(
        self,
        df: pl.DataFrame,
        period: Optional[int] = None,
    ) -> pl.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Period for ATR (defaults to config.atr_period)

        Returns:
            Series with ATR values
        """
        if period is None:
            period = self.config.atr_period

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range calculations
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        # Create temporary DataFrame for max calculation
        tr_df = pl.DataFrame(
            {
                "tr1": tr1,
                "tr2": tr2,
                "tr3": tr3,
            }
        )

        # True Range is the maximum of the three
        true_range = tr_df.select(pl.max_horizontal("tr1", "tr2", "tr3")).to_series()

        # ATR is SMA of True Range
        atr = true_range.rolling_mean(window_size=period)

        return atr

    # ========== ADDITIONAL MOMENTUM INDICATORS ==========

    def calculate_roc(self, df: pl.DataFrame, period: int = 12) -> pl.Series:
        """
        Calculate Rate of Change.

        Args:
            df: DataFrame with 'close' column
            period: ROC period

        Returns:
            Series with ROC values (percentage)
        """
        close = df["close"]
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc.fill_null(0.0)

    def calculate_ppo(
        self,
        df: pl.DataFrame,
        fast: int = 12,
        slow: int = 26,
    ) -> pl.Series:
        """
        Calculate Percentage Price Oscillator.

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period

        Returns:
            Series with PPO values
        """
        ema_fast = self.calculate_ema(df, period=fast)
        ema_slow = self.calculate_ema(df, period=slow)

        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        return ppo.fill_null(0.0)

    def calculate_stochastic(
        self,
        df: pl.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict[str, pl.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period
            d_period: %D smoothing period

        Returns:
            Dictionary with 'stoch_k' and 'stoch_d' series
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Highest high and lowest low over period
        highest_high = high.rolling_max(window_size=k_period)
        lowest_low = low.rolling_min(window_size=k_period)

        # %K
        stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        stoch_k = stoch_k.fill_null(50.0)

        # %D (SMA of %K)
        stoch_d = stoch_k.rolling_mean(window_size=d_period)

        return {
            "stoch_k": stoch_k,
            "stoch_d": stoch_d.fill_null(50.0),
        }

    def calculate_williams_r(
        self,
        df: pl.DataFrame,
        period: int = 14,
    ) -> pl.Series:
        """
        Calculate Williams %R.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period

        Returns:
            Series with Williams %R values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        highest_high = high.rolling_max(window_size=period)
        lowest_low = low.rolling_min(window_size=period)

        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r.fill_null(-50.0)

    # ========== VOLUME INDICATORS ==========

    def calculate_obv(self, df: pl.DataFrame) -> pl.Series:
        """
        Calculate On-Balance Volume.

        Args:
            df: DataFrame with 'close' and 'volume' columns

        Returns:
            Series with OBV values
        """
        close = df["close"]
        volume = df["volume"]

        # Price direction
        direction = (
            pl.when(close > close.shift(1))
            .then(1)
            .otherwise(pl.when(close < close.shift(1)).then(-1).otherwise(0))
        )

        # OBV is cumulative sum of signed volume
        signed_volume = (direction * volume).fill_null(0)
        obv = signed_volume.cum_sum()

        return obv

    def calculate_vwap_deviation(self, df: pl.DataFrame) -> pl.Series:
        """
        Calculate VWAP deviation (price - VWAP) / VWAP.

        Args:
            df: DataFrame with 'close', 'high', 'low', 'volume' columns

        Returns:
            Series with VWAP deviation percentage
        """
        # Typical price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # VWAP calculation (cumulative for the session)
        vwap = (typical_price * df["volume"]).cum_sum() / df["volume"].cum_sum()

        # Deviation from VWAP as percentage
        deviation = ((df["close"] - vwap) / vwap) * 100

        return deviation.fill_null(0.0)

    def calculate_volume_ratio(
        self,
        df: pl.DataFrame,
        period: int = 20,
    ) -> pl.Series:
        """
        Calculate volume ratio (current volume / average volume).

        Args:
            df: DataFrame with 'volume' column
            period: Lookback period for average

        Returns:
            Series with volume ratio
        """
        volume = df["volume"].cast(pl.Float64)
        avg_volume = volume.rolling_mean(window_size=period)

        ratio = volume / avg_volume
        return ratio.fill_null(1.0)

    # ========== VOLATILITY INDICATORS ==========

    def calculate_keltner_channels(
        self,
        df: pl.DataFrame,
        period: int = 20,
        atr_multiplier: float = 2.0,
    ) -> dict[str, pl.Series]:
        """
        Calculate Keltner Channels.

        Args:
            df: DataFrame with OHLC columns
            period: EMA period
            atr_multiplier: ATR multiplier for bands

        Returns:
            Dictionary with 'upper', 'middle', 'lower' series
        """
        middle = self.calculate_ema(df, period=period)
        atr = self.calculate_atr(df, period=period)

        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    def calculate_donchian_channels(
        self,
        df: pl.DataFrame,
        period: int = 20,
    ) -> dict[str, pl.Series]:
        """
        Calculate Donchian Channels.

        Args:
            df: DataFrame with 'high' and 'low' columns
            period: Lookback period

        Returns:
            Dictionary with 'upper', 'middle', 'lower' series
        """
        upper = df["high"].rolling_max(window_size=period)
        lower = df["low"].rolling_min(window_size=period)
        middle = (upper + lower) / 2

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    def calculate_squeeze_pro(
        self,
        df: pl.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_length: int = 12,
        mom_smooth: int = 6,
    ) -> dict[str, pl.Series]:
        """
        Calculate Squeeze Pro (TTM Squeeze Pro) indicator.

        Detects volatility squeeze when Bollinger Bands are inside Keltner Channels,
        indicating low volatility period before potential breakouts.
        Includes momentum component to predict breakout direction.

        Based on research findings: Squeeze_pro is the #1 ranked indicator for swing trading.

        Args:
            df: DataFrame with OHLC columns
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviation multiplier
            kc_period: Keltner Channel period
            kc_mult: Keltner Channel ATR multiplier
            mom_length: Momentum calculation period
            mom_smooth: Momentum smoothing period

        Returns:
            Dictionary with:
            - 'squeeze': Boolean series (True when BB inside KC)
            - 'squeeze_pro': Float series (squeeze intensity: 1.0=high, 0.5=medium, 0.0=none)
            - 'momentum': Momentum histogram for breakout direction
        """
        # Calculate Bollinger Bands
        bb = self.calculate_bollinger_bands(df, period=bb_period, stddev=bb_std)

        # Calculate Keltner Channels
        kc = self.calculate_keltner_channels(df, period=kc_period, atr_multiplier=kc_mult)

        # Squeeze condition: BB upper < KC upper AND BB lower > KC lower
        squeeze_condition = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])

        # Calculate squeeze intensity using multiple KC multipliers
        # Compare BB to KC with different ATR multipliers (1.0, 1.5, 2.0)
        kc_tight = self.calculate_keltner_channels(df, period=kc_period, atr_multiplier=1.0)
        kc_medium = self.calculate_keltner_channels(df, period=kc_period, atr_multiplier=1.5)
        kc_wide = self.calculate_keltner_channels(df, period=kc_period, atr_multiplier=2.0)

        # Squeeze intensity: 1.0 if BB inside tight KC, 0.5 if inside medium, 0.0 if no squeeze
        squeeze_tight = (bb["upper"] < kc_tight["upper"]) & (bb["lower"] > kc_tight["lower"])
        squeeze_medium = (bb["upper"] < kc_medium["upper"]) & (bb["lower"] > kc_medium["lower"])

        squeeze_pro = pl.when(squeeze_tight).then(1.0).when(squeeze_medium).then(0.5).otherwise(0.0)

        # Momentum component: linear regression slope of price change
        close = df["close"]
        # Calculate momentum as price change over mom_length periods
        momentum_raw = close - close.shift(mom_length)
        # Smooth with EMA
        mom_df = pl.DataFrame({"mom": momentum_raw.to_list()})
        alpha = 2.0 / (mom_smooth + 1.0)
        momentum = mom_df["mom"].ewm_mean(alpha=alpha, adjust=False)

        return {
            "squeeze": squeeze_condition.cast(pl.Int8),
            "squeeze_pro": squeeze_pro,
            "momentum": momentum.fill_null(0.0),
        }

    def calculate_natr(
        self,
        df: pl.DataFrame,
        period: int = 14,
    ) -> pl.Series:
        """
        Calculate Normalized ATR (ATR / Close * 100).

        Args:
            df: DataFrame with OHLC columns
            period: ATR period

        Returns:
            Series with NATR values (percentage)
        """
        atr = self.calculate_atr(df, period=period)
        natr = (atr / df["close"]) * 100
        return natr.fill_null(0.0)

    # ========== TREND INDICATORS ==========

    def calculate_adx(
        self,
        df: pl.DataFrame,
        period: int = 14,
    ) -> dict[str, pl.Series]:
        """
        Calculate Average Directional Index.

        Args:
            df: DataFrame with OHLC columns
            period: Smoothing period

        Returns:
            Dictionary with 'adx', 'plus_di', 'minus_di' series
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate +DM and -DM
        plus_dm = (high - high.shift(1)).clip(0, None)
        minus_dm = (low.shift(1) - low).clip(0, None)

        # Zero out when one is larger than the other
        plus_dm = pl.when(plus_dm > minus_dm).then(plus_dm).otherwise(0.0)
        minus_dm = pl.when(minus_dm > plus_dm).then(minus_dm).otherwise(0.0)

        # Calculate ATR
        atr = self.calculate_atr(df, period=period)

        # Smooth +DM and -DM
        alpha = 1.0 / period

        # Create temp df for smoothing - need to evaluate expressions first
        temp_df = df.select([plus_dm.alias("plus_dm"), minus_dm.alias("minus_dm")])
        smooth_plus_dm = temp_df["plus_dm"].ewm_mean(alpha=alpha, adjust=False)
        smooth_minus_dm = temp_df["minus_dm"].ewm_mean(alpha=alpha, adjust=False)

        # Calculate +DI and -DI
        plus_di = (smooth_plus_dm / atr) * 100
        minus_di = (smooth_minus_dm / atr) * 100

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = (di_diff / di_sum.replace({0.0: None})) * 100
        dx = dx.fill_null(0.0)

        # Smooth DX to get ADX - need to evaluate expression first
        dx_df = df.select([dx.alias("dx")])
        adx = dx_df["dx"].ewm_mean(alpha=alpha, adjust=False)

        return {
            "adx": adx.fill_null(0.0),
            "plus_di": plus_di.fill_null(0.0),
            "minus_di": minus_di.fill_null(0.0),
        }

    # ========== FRACTIONAL DIFFERENCING (LOPEZ DE PRADO) ==========

    def _get_weights_ffd(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Get weights for Fixed-width Window Fractional Differentiation.

        Based on "Advances in Financial Machine Learning" by Lopez de Prado.

        Args:
            d: Fractional differencing order (0 < d < 1)
            threshold: Minimum weight threshold

        Returns:
            Array of weights
        """
        weights = [1.0]
        k = 1

        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1

        return np.array(weights[::-1])

    def fractional_diff(
        self,
        series: pl.Series,
        d: float = 0.5,
        threshold: float = 1e-5,
    ) -> pl.Series:
        """
        Apply fractional differencing to a series.

        This preserves memory while achieving stationarity.
        Per Lopez de Prado's "Advances in Financial Machine Learning".

        Args:
            series: Price or indicator series
            d: Differencing order (0.0 = no diff, 1.0 = full diff)
               Typical values: 0.3-0.5 for price series
            threshold: Weight cutoff threshold

        Returns:
            Fractionally differenced series
        """
        weights = self._get_weights_ffd(d, threshold)
        width = len(weights)

        # Convert to numpy for computation
        values = series.to_numpy()
        n = len(values)

        # Apply fractional differencing
        result = np.full(n, np.nan)

        for i in range(width - 1, n):
            result[i] = np.dot(weights, values[i - width + 1 : i + 1])

        return pl.Series(result)

    def apply_fractional_diff_features(
        self,
        df: pl.DataFrame,
        columns: list[str],
        d: float = 0.5,
    ) -> pl.DataFrame:
        """
        Apply fractional differencing to multiple columns.

        Args:
            df: DataFrame with feature columns
            columns: List of columns to difference
            d: Differencing order

        Returns:
            DataFrame with new '_ffd' suffixed columns
        """
        result = df.clone()

        for col in columns:
            if col in df.columns:
                ffd_col = self.fractional_diff(df[col], d=d)
                result = result.with_columns(ffd_col.alias(f"{col}_ffd"))

        return result

    # ========== Z-SCORE NORMALIZATION ==========

    def zscore_normalize(
        self,
        series: pl.Series,
        window: int = 252,
    ) -> pl.Series:
        """
        Calculate rolling z-score normalization.

        Args:
            series: Input series
            window: Rolling window for mean/std

        Returns:
            Z-score normalized series
        """
        mean = series.rolling_mean(window_size=window)
        std = series.rolling_std(window_size=window)

        zscore = (series - mean) / std.replace({0.0: None})
        return zscore.fill_null(0.0)

    def normalize_features(
        self,
        df: pl.DataFrame,
        columns: list[str],
        window: int = 252,
    ) -> pl.DataFrame:
        """
        Apply z-score normalization to multiple columns.

        Args:
            df: DataFrame with feature columns
            columns: List of columns to normalize
            window: Rolling window size

        Returns:
            DataFrame with '_zscore' suffixed columns
        """
        result = df.clone()

        for col in columns:
            if col in df.columns:
                zscore_col = self.zscore_normalize(df[col], window=window)
                result = result.with_columns(zscore_col.alias(f"{col}_zscore"))

        return result

    def calculate_all_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all technical indicators and add to DataFrame.

        Implements comprehensive indicator suite based on research:
        - Momentum: RSI, PPO, ROC, Stochastic, Williams %R
        - Trend: MACD, ADX, EMAs, SMAs
        - Volatility: ATR, NATR, Bollinger Bands, Keltner Channels
        - Volume: OBV, VWAP deviation, Volume ratio

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators added
        """
        result = df.clone()

        # ========== MOVING AVERAGES ==========
        result = result.with_columns(
            [
                self.calculate_sma(df, period=self.config.sma_short).alias("sma_20"),
                self.calculate_sma(df, period=self.config.sma_long).alias("sma_50"),
                self.calculate_sma(df, period=self.config.sma_trend).alias("sma_200"),
                self.calculate_ema(df, period=9).alias("ema_9"),
                self.calculate_ema(df, period=21).alias("ema_21"),
            ]
        )

        # ========== MOMENTUM INDICATORS ==========
        result = result.with_columns(
            [
                self.calculate_rsi(df).alias("rsi"),
                self.calculate_rsi(df, period=63).alias("rsi_63"),  # Research-backed: 63-day RSI
                self.calculate_roc(df, period=12).alias("roc_12"),
                self.calculate_roc(df, period=63).alias("roc_63"),  # Research-backed: 63-day ROC
                self.calculate_ppo(df).alias("ppo"),
                self.calculate_williams_r(df).alias("williams_r"),
            ]
        )

        # Stochastic
        stoch = self.calculate_stochastic(df)
        result = result.with_columns(
            [
                stoch["stoch_k"].alias("stoch_k"),
                stoch["stoch_d"].alias("stoch_d"),
            ]
        )

        # ========== MACD ==========
        macd = self.calculate_macd(df)
        result = result.with_columns(
            [
                macd["macd"].alias("macd"),
                macd["signal"].alias("macd_signal"),
                macd["histogram"].alias("macd_histogram"),
            ]
        )

        # ========== ADX ==========
        adx = self.calculate_adx(df)
        result = result.with_columns(
            [
                adx["adx"].alias("adx"),
                adx["plus_di"].alias("plus_di"),
                adx["minus_di"].alias("minus_di"),
            ]
        )

        # ========== BOLLINGER BANDS ==========
        bb = self.calculate_bollinger_bands(df)
        result = result.with_columns(
            [
                bb["upper"].alias("bb_upper"),
                bb["middle"].alias("bb_middle"),
                bb["lower"].alias("bb_lower"),
            ]
        )

        # Bollinger Band %B (where price is within bands)
        bb_pct_b = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
        result = result.with_columns(
            [
                bb_pct_b.fill_null(0.5).alias("bb_pct_b"),
            ]
        )

        # ========== KELTNER CHANNELS ==========
        kc = self.calculate_keltner_channels(df)
        result = result.with_columns(
            [
                kc["upper"].alias("kc_upper"),
                kc["middle"].alias("kc_middle"),
                kc["lower"].alias("kc_lower"),
            ]
        )

        # ========== SQUEEZE PRO (Research #1 Indicator) ==========
        squeeze = self.calculate_squeeze_pro(df)
        result = result.with_columns(
            [
                squeeze["squeeze"].alias("squeeze"),
                squeeze["squeeze_pro"].alias("squeeze_pro"),
                squeeze["momentum"].alias("squeeze_momentum"),
            ]
        )

        # ========== VOLATILITY ==========
        result = result.with_columns(
            [
                self.calculate_atr(df).alias("atr"),
                self.calculate_natr(df).alias("natr"),
            ]
        )

        # ========== VOLUME INDICATORS ==========
        result = result.with_columns(
            [
                self.calculate_obv(df).alias("obv"),
                self.calculate_vwap_deviation(df).alias("vwap_dev"),
                self.calculate_volume_ratio(df).alias("volume_ratio"),
            ]
        )

        # ========== PRICE PATTERNS ==========
        # Price position relative to moving averages
        result = result.with_columns(
            [
                ((df["close"] - result["sma_20"]) / result["sma_20"] * 100).alias(
                    "price_sma20_pct"
                ),
                ((df["close"] - result["sma_50"]) / result["sma_50"] * 100).alias(
                    "price_sma50_pct"
                ),
                ((df["close"] - result["sma_200"]) / result["sma_200"] * 100).alias(
                    "price_sma200_pct"
                ),
            ]
        )

        # Moving average crossover signals
        result = result.with_columns(
            [
                (result["sma_20"] > result["sma_50"]).cast(pl.Int8).alias("sma_20_50_cross"),
                (result["ema_9"] > result["ema_21"]).cast(pl.Int8).alias("ema_9_21_cross"),
            ]
        )

        # ========== RETURNS ==========
        result = result.with_columns(
            [
                df["close"].pct_change().fill_null(0.0).alias("return_1d"),
                df["close"].pct_change(n=5).fill_null(0.0).alias("return_5d"),
                df["close"].pct_change(n=20).fill_null(0.0).alias("return_20d"),
            ]
        )

        logger.debug(
            f"Calculated {len(result.columns) - len(df.columns)} technical indicators for {len(result)} rows"
        )
        return result

    def calculate_ml_features(
        self,
        df: pl.DataFrame,
        apply_ffd: bool = True,
        apply_zscore: bool = True,
        ffd_d: float = 0.5,
        zscore_window: int = 252,
    ) -> pl.DataFrame:
        """
        Calculate ML-ready features with optional transformations.

        This is the primary feature generation method for the ML pipeline.
        Applies all indicators, fractional differencing, and normalization.

        Args:
            df: DataFrame with OHLCV data
            apply_ffd: Apply fractional differencing to price-based features
            apply_zscore: Apply z-score normalization
            ffd_d: Fractional differencing order
            zscore_window: Z-score rolling window

        Returns:
            DataFrame with ML-ready features
        """
        # Calculate all base indicators
        result = self.calculate_all_indicators(df)

        # Price-based columns that benefit from fractional differencing
        price_based_cols = ["close", "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "obv"]

        if apply_ffd:
            result = self.apply_fractional_diff_features(result, columns=price_based_cols, d=ffd_d)
            logger.debug(f"Applied fractional differencing (d={ffd_d})")

        # Columns that benefit from z-score normalization
        zscore_cols = [
            "rsi",
            "rsi_63",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "macd",
            "macd_histogram",
            "ppo",
            "roc_12",
            "roc_63",
            "adx",
            "atr",
            "natr",
            "vwap_dev",
            "volume_ratio",
            "bb_pct_b",
            "price_sma20_pct",
            "price_sma50_pct",
            "price_sma200_pct",
            "squeeze_pro",
            "squeeze_momentum",
        ]

        if apply_zscore:
            result = self.normalize_features(result, columns=zscore_cols, window=zscore_window)
            logger.debug(f"Applied z-score normalization (window={zscore_window})")

        # Drop rows with NaN values (warm-up period)
        # Only drop rows where critical features are null, not all features
        # This prevents dropping all rows when some indicators need more data than available
        initial_rows = len(result)

        # Critical features that must be non-null (core indicators)
        critical_features = ["close", "rsi", "macd", "atr"]
        available_critical = [f for f in critical_features if f in result.columns]

        if available_critical:
            result = result.drop_nulls(subset=available_critical)
        else:
            # Fallback: drop rows where all features are null
            result = result.drop_nulls(how="all")

        dropped_rows = initial_rows - len(result)

        if dropped_rows > 0:
            logger.debug(f"Dropped {dropped_rows} rows with null values (warm-up period)")

        logger.info(f"Generated {len(result.columns)} ML features for {len(result)} rows")
        return result

    def get_feature_names(self, include_ffd: bool = True, include_zscore: bool = True) -> list[str]:
        """
        Get list of feature names that will be generated.

        Args:
            include_ffd: Include fractionally differenced features
            include_zscore: Include z-score normalized features

        Returns:
            List of feature column names
        """
        base_features = [
            # Moving averages
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_9",
            "ema_21",
            # Momentum
            "rsi",
            "rsi_63",
            "roc_12",
            "roc_63",
            "ppo",
            "williams_r",
            "stoch_k",
            "stoch_d",
            # Squeeze Pro (Research #1)
            "squeeze",
            "squeeze_pro",
            "squeeze_momentum",
            # MACD
            "macd",
            "macd_signal",
            "macd_histogram",
            # ADX
            "adx",
            "plus_di",
            "minus_di",
            # Bollinger Bands
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_pct_b",
            # Keltner Channels
            "kc_upper",
            "kc_middle",
            "kc_lower",
            # Volatility
            "atr",
            "natr",
            # Volume
            "obv",
            "vwap_dev",
            "volume_ratio",
            # Price patterns
            "price_sma20_pct",
            "price_sma50_pct",
            "price_sma200_pct",
            "sma_20_50_cross",
            "ema_9_21_cross",
            # Returns
            "return_1d",
            "return_5d",
            "return_20d",
        ]

        features = base_features.copy()

        if include_ffd:
            ffd_cols = ["close", "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "obv"]
            features.extend([f"{col}_ffd" for col in ffd_cols])

        if include_zscore:
            zscore_cols = [
                "rsi",
                "stoch_k",
                "stoch_d",
                "williams_r",
                "macd",
                "macd_histogram",
                "ppo",
                "roc_12",
                "adx",
                "atr",
                "natr",
                "vwap_dev",
                "volume_ratio",
                "bb_pct_b",
                "price_sma20_pct",
                "price_sma50_pct",
                "price_sma200_pct",
            ]
            features.extend([f"{col}_zscore" for col in zscore_cols])

        return features
