"""Multi-signal confluence layer with regime-aware weighting.

Research-backed implementation based on:
- "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen, 2013)
- "Can Machines Build Better Stock Portfolios?" (AQR)
- Forecast combination theory (Bates & Granger)

Key patterns:
- Z-score normalization for signal combination
- IC-weighted and equal-weight combinations
- Regime-aware weighting based on market conditions
- Ensemble aggregation with meta-model option
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    NEUTRAL = "neutral"


class SignalDirection(Enum):
    """Signal direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class ConfluenceSignal:
    """Result of confluence analysis."""

    direction: SignalDirection
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    technical_score: float  # -1 to 1
    fundamental_score: float  # -1 to 1
    sentiment_score: float  # -1 to 1
    confluence_score: float  # Combined score
    regime: MarketRegime
    individual_signals: dict[str, float]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "technical_score": self.technical_score,
            "fundamental_score": self.fundamental_score,
            "sentiment_score": self.sentiment_score,
            "confluence_score": self.confluence_score,
            "regime": self.regime.value,
            "individual_signals": self.individual_signals,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfluenceLayer:
    """Multi-signal confluence engine for trading decisions.

    Implements institutional-grade signal combination:
    - Technical, fundamental, and sentiment signal integration
    - Z-score normalization for comparable signals
    - IC-weighted combination based on historical performance
    - Regime-aware dynamic weighting
    - Confidence calibration
    """

    # Default signal weights (can be adjusted via config or IC)
    DEFAULT_WEIGHTS = {
        "technical": 0.40,
        "sentiment": 0.35,
        "fundamental": 0.25,
    }

    # Regime-specific weight adjustments
    # Based on research showing technical works better in high-sentiment periods
    REGIME_WEIGHTS = {
        MarketRegime.BULLISH: {
            "technical": 0.35,
            "sentiment": 0.40,
            "fundamental": 0.25,
        },
        MarketRegime.BEARISH: {
            "technical": 0.30,
            "sentiment": 0.30,
            "fundamental": 0.40,
        },
        MarketRegime.HIGH_VOLATILITY: {
            "technical": 0.50,
            "sentiment": 0.30,
            "fundamental": 0.20,
        },
        MarketRegime.LOW_VOLATILITY: {
            "technical": 0.35,
            "sentiment": 0.30,
            "fundamental": 0.35,
        },
        MarketRegime.TRENDING: {
            "technical": 0.50,
            "sentiment": 0.30,
            "fundamental": 0.20,
        },
        MarketRegime.MEAN_REVERTING: {
            "technical": 0.30,
            "sentiment": 0.35,
            "fundamental": 0.35,
        },
        MarketRegime.NEUTRAL: DEFAULT_WEIGHTS,
    }

    def __init__(
        self,
        use_regime_weights: bool = True,
        min_confluence_threshold: float = 0.3,
        signal_lookback: int = 20,
    ) -> None:
        """
        Initialize confluence layer.

        Args:
            use_regime_weights: Whether to use regime-aware weighting
            min_confluence_threshold: Minimum score for signal generation
            signal_lookback: Lookback periods for signal normalization
        """
        self.use_regime_weights = use_regime_weights
        self.min_confluence_threshold = min_confluence_threshold
        self.signal_lookback = signal_lookback

        # IC tracking for adaptive weighting
        self.ic_history: dict[str, list[float]] = {
            "technical": [],
            "sentiment": [],
            "fundamental": [],
        }

        logger.info(
            f"ConfluenceLayer initialized: regime_weights={use_regime_weights}, "
            f"threshold={min_confluence_threshold}"
        )

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: float,
        trend_strength: float,
        sentiment_level: float,
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            returns: Recent returns array
            volatility: Current volatility (e.g., ATR/price)
            trend_strength: Trend indicator (e.g., ADX)
            sentiment_level: Aggregate sentiment (-1 to 1)

        Returns:
            Detected market regime
        """
        # Calculate return momentum
        if len(returns) >= 20:
            return_20d = np.sum(returns[-20:])
        else:
            return_20d = np.sum(returns)

        # Volatility regime (using typical thresholds)
        high_vol_threshold = 0.02  # 2% daily vol
        low_vol_threshold = 0.01  # 1% daily vol

        if volatility > high_vol_threshold:
            vol_regime = "high"
        elif volatility < low_vol_threshold:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        # Trend regime (ADX-based)
        trend_threshold = 25
        is_trending = trend_strength > trend_threshold

        # Direction regime
        if return_20d > 0.05:  # 5% gain
            direction = "bullish"
        elif return_20d < -0.05:  # 5% loss
            direction = "bearish"
        else:
            direction = "neutral"

        # Combine factors into regime
        if vol_regime == "high":
            return MarketRegime.HIGH_VOLATILITY
        elif vol_regime == "low":
            return MarketRegime.LOW_VOLATILITY
        elif is_trending:
            return MarketRegime.TRENDING
        elif direction == "bullish":
            return MarketRegime.BULLISH
        elif direction == "bearish":
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def normalize_signal(
        self,
        signal: float,
        signal_history: Optional[list[float]] = None,
    ) -> float:
        """
        Z-score normalize a signal.

        Args:
            signal: Raw signal value
            signal_history: Historical signals for normalization

        Returns:
            Normalized signal (-1 to 1 range, clipped)
        """
        if signal_history is None or len(signal_history) < 5:
            # Simple clip if no history
            return np.clip(signal, -1, 1)

        mean = np.mean(signal_history)
        std = np.std(signal_history)

        if std == 0:
            return 0.0

        zscore = (signal - mean) / std

        # Clip to [-3, 3] then scale to [-1, 1]
        zscore = np.clip(zscore, -3, 3) / 3

        return zscore

    def combine_signals(
        self,
        technical_score: float,
        sentiment_score: float,
        fundamental_score: float,
        regime: MarketRegime,
        use_ic_weights: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """
        Combine multiple signals into a single confluence score.

        Args:
            technical_score: Technical signal (-1 to 1)
            sentiment_score: Sentiment signal (-1 to 1)
            fundamental_score: Fundamental signal (-1 to 1)
            regime: Current market regime
            use_ic_weights: Use IC-based adaptive weights

        Returns:
            Tuple of (combined_score, weights_used)
        """
        # Get appropriate weights
        if use_ic_weights and self._has_sufficient_ic_history():
            weights = self._calculate_ic_weights()
        elif self.use_regime_weights:
            weights = self.REGIME_WEIGHTS.get(regime, self.DEFAULT_WEIGHTS)
        else:
            weights = self.DEFAULT_WEIGHTS

        # Weighted combination
        combined = (
            weights["technical"] * technical_score
            + weights["sentiment"] * sentiment_score
            + weights["fundamental"] * fundamental_score
        )

        return combined, weights

    def _has_sufficient_ic_history(self, min_periods: int = 20) -> bool:
        """Check if we have enough IC history for adaptive weights."""
        return all(len(self.ic_history[key]) >= min_periods for key in self.ic_history)

    def _calculate_ic_weights(self) -> dict[str, float]:
        """Calculate IC-based weights from historical performance."""
        # Calculate mean IC for each signal type
        mean_ics = {}
        for key, history in self.ic_history.items():
            if history:
                mean_ics[key] = abs(np.mean(history))  # Use absolute IC
            else:
                mean_ics[key] = 0.33

        # Normalize to sum to 1
        total_ic = sum(mean_ics.values())
        if total_ic == 0:
            return self.DEFAULT_WEIGHTS

        weights = {key: ic / total_ic for key, ic in mean_ics.items()}

        logger.debug(f"IC-based weights: {weights}")
        return weights

    def record_ic(
        self,
        signal_type: str,
        predicted: float,
        actual: float,
    ) -> None:
        """
        Record information coefficient for adaptive weighting.

        IC = correlation between predicted and actual returns.

        Args:
            signal_type: Type of signal (technical, sentiment, fundamental)
            predicted: Predicted direction/magnitude
            actual: Actual return
        """
        if signal_type in self.ic_history:
            # Simple IC proxy: sign agreement
            ic = 1.0 if (predicted > 0) == (actual > 0) else -1.0
            self.ic_history[signal_type].append(ic)

            # Keep limited history
            if len(self.ic_history[signal_type]) > 100:
                self.ic_history[signal_type] = self.ic_history[signal_type][-100:]

    def generate_confluence_signal(
        self,
        technical_signals: dict[str, float],
        sentiment_data: dict[str, Any],
        fundamental_data: Optional[dict[str, float]] = None,
        market_data: Optional[dict[str, float]] = None,
    ) -> ConfluenceSignal:
        """
        Generate a confluence trading signal.

        Args:
            technical_signals: Dictionary of technical indicator values
            sentiment_data: Sentiment analysis results
            fundamental_data: Optional fundamental factor values
            market_data: Optional market data for regime detection

        Returns:
            ConfluenceSignal with combined analysis
        """
        # Calculate technical score from signals
        technical_score = self._calculate_technical_score(technical_signals)

        # Extract sentiment score
        sentiment_score = self._extract_sentiment_score(sentiment_data)

        # Calculate fundamental score
        fundamental_score = self._calculate_fundamental_score(fundamental_data)

        # Detect regime if market data provided
        if market_data:
            regime = self.detect_regime(
                returns=np.array(market_data.get("returns", [0])),
                volatility=market_data.get("volatility", 0.015),
                trend_strength=market_data.get("adx", 20),
                sentiment_level=sentiment_score,
            )
        else:
            regime = MarketRegime.NEUTRAL

        # Combine signals
        confluence_score, weights = self.combine_signals(
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            fundamental_score=fundamental_score,
            regime=regime,
        )

        # Determine direction and strength
        if confluence_score > self.min_confluence_threshold:
            direction = SignalDirection.LONG
            strength = min(1.0, confluence_score)
        elif confluence_score < -self.min_confluence_threshold:
            direction = SignalDirection.SHORT
            strength = min(1.0, abs(confluence_score))
        else:
            direction = SignalDirection.NEUTRAL
            strength = abs(confluence_score)

        # Calculate confidence based on signal agreement
        confidence = self._calculate_confidence(
            technical_score,
            sentiment_score,
            fundamental_score,
        )

        signal = ConfluenceSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            confluence_score=confluence_score,
            regime=regime,
            individual_signals={
                **technical_signals,
                "sentiment": sentiment_score,
                "fundamental": fundamental_score,
            },
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"Confluence signal: {direction.value} "
            f"(strength={strength:.2f}, conf={confidence:.2f}, regime={regime.value})"
        )

        return signal

    def _calculate_technical_score(
        self,
        signals: dict[str, float],
    ) -> float:
        """
        Calculate aggregate technical score from individual indicators.

        Args:
            signals: Dictionary of indicator values

        Returns:
            Technical score (-1 to 1)
        """
        if not signals:
            return 0.0

        # Indicator weights (based on research findings)
        indicator_weights = {
            "rsi": 0.15,
            "macd": 0.15,
            "macd_histogram": 0.10,
            "bb_pct_b": 0.10,
            "adx": 0.10,
            "stoch_k": 0.10,
            "momentum": 0.10,
            "trend": 0.20,
        }

        score = 0.0
        total_weight = 0.0

        for indicator, weight in indicator_weights.items():
            if indicator in signals:
                value = signals[indicator]

                # Normalize indicator to -1 to 1 range
                if indicator == "rsi":
                    # RSI: 0-100, >70 overbought, <30 oversold
                    normalized = (value - 50) / 50  # -1 to 1
                elif indicator == "bb_pct_b":
                    # BB %B: 0-1 typically
                    normalized = (value - 0.5) * 2  # -1 to 1
                elif indicator == "stoch_k":
                    normalized = (value - 50) / 50
                elif indicator == "adx":
                    # ADX: 0-100, higher = stronger trend
                    normalized = (value - 25) / 25  # Normalize around 25
                    normalized = np.clip(normalized, -1, 1)
                else:
                    # Assume already normalized or use as-is
                    normalized = np.clip(value, -1, 1)

                score += weight * normalized
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        return np.clip(score, -1, 1)

    def _extract_sentiment_score(
        self,
        sentiment_data: dict[str, Any],
    ) -> float:
        """
        Extract normalized sentiment score.

        Args:
            sentiment_data: Sentiment analysis results

        Returns:
            Sentiment score (-1 to 1)
        """
        if not sentiment_data:
            return 0.0

        # Handle different sentiment data formats
        if "aggregated_score" in sentiment_data:
            return np.clip(sentiment_data["aggregated_score"], -1, 1)
        elif "score" in sentiment_data:
            return np.clip(sentiment_data["score"], -1, 1)
        elif "sentiment_score" in sentiment_data:
            return np.clip(sentiment_data["sentiment_score"], -1, 1)

        return 0.0

    def _calculate_fundamental_score(
        self,
        fundamental_data: Optional[dict[str, float]],
    ) -> float:
        """
        Calculate fundamental factor score.

        Based on Quality Minus Junk and Value factors.

        Args:
            fundamental_data: Fundamental metrics

        Returns:
            Fundamental score (-1 to 1)
        """
        if not fundamental_data:
            return 0.0

        # Factor weights (based on FF5 + Quality research)
        factor_weights = {
            "value": 0.25,  # B/M, E/P
            "quality": 0.25,  # ROE, margins
            "momentum": 0.20,  # 12-1 month return
            "growth": 0.15,  # Revenue, EPS growth
            "low_vol": 0.15,  # Volatility factor
        }

        score = 0.0
        total_weight = 0.0

        for factor, weight in factor_weights.items():
            if factor in fundamental_data:
                value = fundamental_data[factor]
                score += weight * np.clip(value, -1, 1)
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        return np.clip(score, -1, 1)

    def _calculate_confidence(
        self,
        technical: float,
        sentiment: float,
        fundamental: float,
    ) -> float:
        """
        Calculate confidence based on signal agreement.

        Higher confidence when signals agree in direction.

        Args:
            technical: Technical score
            sentiment: Sentiment score
            fundamental: Fundamental score

        Returns:
            Confidence level (0 to 1)
        """
        signals = [technical, sentiment, fundamental]

        # Check directional agreement
        positive = sum(1 for s in signals if s > 0.1)
        negative = sum(1 for s in signals if s < -0.1)
        neutral = sum(1 for s in signals if abs(s) <= 0.1)

        # Full agreement = high confidence
        if positive == 3 or negative == 3:
            base_confidence = 0.9
        # Two agree, one neutral
        elif (positive == 2 and neutral == 1) or (negative == 2 and neutral == 1):
            base_confidence = 0.75
        # Two agree, one disagrees
        elif positive == 2 or negative == 2:
            base_confidence = 0.6
        # Mixed signals
        else:
            base_confidence = 0.4

        # Adjust by signal strength
        avg_strength = np.mean([abs(s) for s in signals])
        confidence = base_confidence * (0.5 + 0.5 * avg_strength)

        return np.clip(confidence, 0, 1)

    def batch_generate_signals(
        self,
        technical_df: pl.DataFrame,
        sentiment_scores: list[float],
        fundamental_scores: Optional[list[float]] = None,
    ) -> pl.DataFrame:
        """
        Generate confluence signals for multiple timestamps.

        Args:
            technical_df: DataFrame with technical indicators
            sentiment_scores: List of sentiment scores
            fundamental_scores: Optional list of fundamental scores

        Returns:
            DataFrame with confluence signals
        """
        n_rows = len(technical_df)

        if fundamental_scores is None:
            fundamental_scores = [0.0] * n_rows

        # Ensure lengths match
        sentiment_scores = sentiment_scores[:n_rows] + [0.0] * max(
            0, n_rows - len(sentiment_scores)
        )
        fundamental_scores = fundamental_scores[:n_rows] + [0.0] * max(
            0, n_rows - len(fundamental_scores)
        )

        results = []

        for i in range(n_rows):
            row = technical_df.row(i, named=True)

            # Extract technical signals from row
            tech_signals = {
                k: v
                for k, v in row.items()
                if k in ["rsi", "macd", "macd_histogram", "bb_pct_b", "adx", "stoch_k"]
            }

            signal = self.generate_confluence_signal(
                technical_signals=tech_signals,
                sentiment_data={"score": sentiment_scores[i]},
                fundamental_data={"value": fundamental_scores[i]},
            )

            results.append(
                {
                    "confluence_score": signal.confluence_score,
                    "direction": signal.direction.value,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "technical_score": signal.technical_score,
                    "sentiment_score": signal.sentiment_score,
                    "fundamental_score": signal.fundamental_score,
                    "regime": signal.regime.value,
                }
            )

        return pl.DataFrame(results)
