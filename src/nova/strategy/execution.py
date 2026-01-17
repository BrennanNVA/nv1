"""Trade signal generation requiring Technical + Fundamental + Sentiment agreement.

Now uses institutional-grade ConfluenceLayer for regime-aware signal combination.
Includes Alpaca API integration for order execution.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

from ..core.config import DataConfig
from ..features.sentiment import SentimentAnalyzer
from ..models.predictor import ModelPredictor
from .confluence import ConfluenceLayer, ConfluenceSignal
from .execution_cost import ExecutionCostModel, OrderType

logger = logging.getLogger(__name__)

# Lazy import for Alpaca trading client
try:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

    ALPACA_TRADING_AVAILABLE = True
except ImportError:
    ALPACA_TRADING_AVAILABLE = False
    logger.warning("alpaca-py trading client not available, order execution disabled")


class ExecutionEngine:
    """Execute trading strategy with multi-factor confirmation using ConfluenceLayer.

    Includes Alpaca API integration for order execution and position tracking.
    """

    def __init__(
        self,
        technical_predictor: Optional[ModelPredictor] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        confluence_layer: Optional[ConfluenceLayer] = None,
        data_config: Optional[DataConfig] = None,
        execution_cost_model: Optional[ExecutionCostModel] = None,
    ) -> None:
        """
        Initialize execution engine.

        Args:
            technical_predictor: Technical analysis model predictor
            sentiment_analyzer: Sentiment analyzer
            confluence_layer: Optional confluence layer (creates default if None)
            data_config: Data configuration for Alpaca credentials
            execution_cost_model: Optional execution cost model for cost estimation
        """
        self.technical_predictor = technical_predictor
        self.sentiment_analyzer = sentiment_analyzer
        self.confluence_layer = confluence_layer or ConfluenceLayer()
        self.execution_cost_model = execution_cost_model or ExecutionCostModel()
        self.trading_client: Optional[TradingClient] = None

        # Initialize Alpaca trading client if credentials available
        if ALPACA_TRADING_AVAILABLE and data_config:
            if data_config.alpaca_api_key and data_config.alpaca_secret_key:
                try:
                    self.trading_client = TradingClient(
                        api_key=data_config.alpaca_api_key,
                        secret_key=data_config.alpaca_secret_key,
                        sandbox=False,  # Pro API
                    )
                    logger.info("Alpaca TradingClient initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Alpaca TradingClient: {e}")
            else:
                logger.warning("Alpaca credentials not configured - order execution disabled")
        else:
            logger.info("Alpaca trading not available - order execution disabled")

        logger.info("ExecutionEngine initialized with ConfluenceLayer and ExecutionCostModel")

    async def generate_signal(
        self,
        technical_features: pl.DataFrame,
        sentiment_text: Optional[str] = None,
        fundamental_data: Optional[dict] = None,
    ) -> dict:
        """
        Generate trading signal with multi-factor confirmation.

        Args:
            technical_features: DataFrame with technical indicators
            sentiment_text: Optional news/sentiment text
            fundamental_data: Optional fundamental data dictionary

        Returns:
            Dictionary with signal details:
            {
                'signal': 1 (buy), -1 (sell), or 0 (hold),
                'confidence': float,
                'technical': float,
                'sentiment': float,
                'fundamental': Optional[float],
                'reasons': list[str]
            }
        """
        signals = {
            "signal": 0,
            "confidence": 0.0,
            "technical": 0.0,
            "sentiment": 0.0,
            "fundamental": None,
            "reasons": [],
        }

        # Technical analysis
        if self.technical_predictor and not technical_features.is_empty():
            try:
                # Get latest row for prediction
                latest = technical_features.tail(1)

                # Select feature columns (exclude target and metadata)
                feature_cols = [
                    col for col in latest.columns if col not in ["symbol", "timestamp", "target"]
                ]
                features_df = latest.select(feature_cols)

                # Make prediction
                pred_df = self.technical_predictor.predict_with_confidence(features_df)

                technical_pred = pred_df["prediction"][0]
                technical_conf = pred_df["confidence"][0]

                # Convert to signal: 1 = buy, 0 = sell/hold
                signals["technical"] = float(technical_pred) * 2 - 1  # 0->-1, 1->1
                signals["confidence"] = technical_conf

                if technical_pred == 1:
                    signals["reasons"].append("Technical model suggests BUY")
                else:
                    signals["reasons"].append("Technical model suggests SELL/HOLD")
            except Exception as e:
                logger.error(f"Technical prediction failed: {e}")
                signals["reasons"].append(f"Technical analysis error: {e}")

        # Sentiment analysis
        if self.sentiment_analyzer and sentiment_text:
            try:
                sentiment_result = await self.sentiment_analyzer.analyze_text(sentiment_text)
                sentiment_score = sentiment_result.get("score", 0.0)
                sentiment_class = sentiment_result.get("classification", "neutral")

                signals["sentiment"] = sentiment_score

                if sentiment_class == "bullish":
                    signals["reasons"].append("Sentiment analysis is BULLISH")
                elif sentiment_class == "bearish":
                    signals["reasons"].append("Sentiment analysis is BEARISH")
                else:
                    signals["reasons"].append("Sentiment analysis is NEUTRAL")
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                signals["reasons"].append(f"Sentiment analysis error: {e}")

        # Fundamental analysis (placeholder - to be implemented)
        if fundamental_data:
            # TODO: Implement fundamental analysis scoring
            signals["fundamental"] = None
            signals["reasons"].append("Fundamental analysis not yet implemented")

        # Combine signals - require agreement
        # Technical and Sentiment must both agree for trade signal
        technical_score = signals["technical"]
        sentiment_score = signals["sentiment"]

        # Both must be positive for buy, both negative for sell
        if technical_score > 0 and sentiment_score > 0.1:
            signals["signal"] = 1  # BUY
            signals["confidence"] = min(signals["confidence"], abs(sentiment_score))
        elif technical_score < 0 and sentiment_score < -0.1:
            signals["signal"] = -1  # SELL
            signals["confidence"] = min(signals["confidence"], abs(sentiment_score))
        else:
            signals["signal"] = 0  # HOLD (no agreement)
            signals["reasons"].append("No agreement between factors - HOLD")

        logger.debug(
            f"Signal generated: {signals['signal']} "
            f"(technical: {technical_score:.2f}, "
            f"sentiment: {sentiment_score:.2f}, "
            f"confidence: {signals['confidence']:.2f})"
        )

        return signals

    def set_technical_predictor(self, predictor: ModelPredictor) -> None:
        """Set technical analysis predictor."""
        self.technical_predictor = predictor
        logger.info("Technical predictor set")

    def set_sentiment_analyzer(self, analyzer: SentimentAnalyzer) -> None:
        """Set sentiment analyzer."""
        self.sentiment_analyzer = analyzer
        logger.info("Sentiment analyzer set")

    async def generate_confluence_signal(
        self,
        symbol: str,
        technical_df: pl.DataFrame,
        sentiment_data: Optional[dict[str, Any]] = None,
        fundamental_data: Optional[dict[str, float]] = None,
    ) -> ConfluenceSignal:
        """
        Generate signal using institutional ConfluenceLayer.

        This is the recommended method for signal generation as it uses
        regime-aware weighting and proper signal combination.

        Args:
            symbol: Stock symbol
            technical_df: DataFrame with technical indicators (latest row used)
            sentiment_data: Optional sentiment analysis results
            fundamental_data: Optional fundamental metrics

        Returns:
            ConfluenceSignal with complete analysis
        """
        if technical_df.is_empty():
            raise ValueError("Technical DataFrame is empty")

        latest = technical_df.tail(1)

        # Extract technical signals from latest row
        technical_signals = {}
        for col in [
            "rsi",
            "macd",
            "macd_histogram",
            "bb_pct_b",
            "adx",
            "stoch_k",
            "williams_r",
            "ppo",
        ]:
            if col in latest.columns:
                value = float(latest[col][0])
                # Normalize to -1 to 1 range for some indicators
                if col == "rsi":
                    technical_signals[col] = (value - 50) / 50  # RSI 0-100 -> -1 to 1
                elif col == "bb_pct_b":
                    technical_signals[col] = (value - 0.5) * 2  # BB %B 0-1 -> -1 to 1
                elif col == "stoch_k":
                    technical_signals[col] = (value - 50) / 50
                else:
                    technical_signals[col] = value

        # Get model prediction if available
        if self.technical_predictor:
            try:
                feature_names = [
                    col
                    for col in latest.columns
                    if col not in ["symbol", "timestamp", "label", "target"]
                ]
                if feature_names:
                    pred_df = self.technical_predictor.predict_with_confidence(
                        latest.select(feature_names)
                    )
                    technical_signals["model_prediction"] = float(pred_df["prediction"][0])
                    technical_signals["model_confidence"] = float(pred_df["confidence"][0])
            except Exception as e:
                logger.debug(f"Model prediction failed: {e}")

        # Prepare market data for regime detection
        returns = (
            technical_df["close"].pct_change().drop_nulls().to_numpy()[-20:]
            if len(technical_df) >= 20
            else np.array([0])
        )
        volatility = (
            float(technical_df["close"].std() / technical_df["close"].mean())
            if len(technical_df) > 0
            else 0.015
        )
        trend_strength = float(latest["adx"][0]) if "adx" in latest.columns else 20.0

        market_data = {
            "returns": returns.tolist(),
            "volatility": volatility,
            "adx": trend_strength,
        }

        # Generate confluence signal
        confluence_signal = self.confluence_layer.generate_confluence_signal(
            technical_signals=technical_signals,
            sentiment_data=sentiment_data or {},
            fundamental_data=fundamental_data,
            market_data=market_data,
        )

        logger.info(
            f"Confluence signal for {symbol}: {confluence_signal.direction.value} "
            f"(strength={confluence_signal.strength:.2f}, confidence={confluence_signal.confidence:.2f}, "
            f"regime={confluence_signal.regime.value})"
        )

        return confluence_signal

    def set_confluence_layer(self, confluence_layer: ConfluenceLayer) -> None:
        """Set confluence layer."""
        self.confluence_layer = confluence_layer
        logger.info("Confluence layer set")

    # ========== ALPACA ORDER EXECUTION ==========

    def estimate_execution_cost(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        avg_daily_volume: float,
        urgency: float = 0.5,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        """
        Estimate execution cost before placing order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            current_price: Current market price
            avg_daily_volume: Average daily volume
            urgency: Execution urgency (0-1)
            order_type: Order type

        Returns:
            Dictionary with cost estimate
        """
        cost = self.execution_cost_model.estimate_cost(
            order_size=float(quantity),
            current_price=current_price,
            avg_daily_volume=avg_daily_volume,
            urgency=urgency,
            order_type=order_type,
        )

        return {
            "estimated_cost": cost.total_cost,
            "slippage": cost.slippage,
            "market_impact": cost.market_impact,
            "commission": cost.commission,
            "spread_cost": cost.spread_cost,
            "participation_rate": cost.participation_rate,
            "cost_as_pct_of_value": (
                (cost.total_cost / cost.order_value * 100) if cost.order_value > 0 else 0.0
            ),
        }

    async def execute_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # "buy" or "sell"
        order_type: str = "market",
        limit_price: Optional[float] = None,
        current_price: Optional[float] = None,
        avg_daily_volume: Optional[float] = None,
        urgency: float = 0.5,
    ) -> dict[str, Any]:
        """
        Execute order via Alpaca API with execution cost estimation.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "buy" or "sell"
            order_type: "market" or "limit"
            limit_price: Required if order_type is "limit"
            current_price: Current market price (for cost estimation)
            avg_daily_volume: Average daily volume (for cost estimation)
            urgency: Execution urgency (0-1, for cost estimation)

        Returns:
            Dictionary with order details, status, and cost estimate
        """
        if not self.trading_client:
            raise RuntimeError("Alpaca TradingClient not initialized")

        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        try:
            # Convert side string to OrderSide enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Create order request
            if order_type.lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise ValueError("limit_price required for limit orders")
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
            else:
                raise ValueError(f"Unsupported order_type: {order_type}")

            # Estimate execution cost if price and volume available
            cost_estimate = None
            if current_price is not None and avg_daily_volume is not None:
                order_type_enum = (
                    OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT
                )
                cost_estimate = self.estimate_execution_cost(
                    symbol=symbol,
                    quantity=quantity,
                    current_price=current_price,
                    avg_daily_volume=avg_daily_volume,
                    urgency=urgency,
                    order_type=order_type_enum,
                )

            # Submit order
            order = self.trading_client.submit_order(order_request)

            result = {
                "success": True,
                "order_id": str(order.id),
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "order_type": order_type,
                "status": (
                    order.status.value if hasattr(order.status, "value") else str(order.status)
                ),
                "submitted_at": (
                    order.submitted_at.isoformat()
                    if hasattr(order, "submitted_at") and order.submitted_at
                    else datetime.utcnow().isoformat()
                ),
            }

            # Add cost estimate if available
            if cost_estimate:
                result["execution_cost_estimate"] = cost_estimate

            logger.info(
                f"Order submitted: {side.upper()} {quantity} {symbol} "
                f"(order_id: {order.id}, type: {order_type})"
            )

            return result

        except APIError as e:
            logger.error(f"Alpaca API error executing order: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
            }
        except Exception as e:
            logger.error(f"Error executing order: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
            }

    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current positions from Alpaca.

        Returns:
            List of position dictionaries
        """
        if not self.trading_client:
            return []

        try:
            positions = self.trading_client.get_all_positions()

            result = []
            for pos in positions:
                result.append(
                    {
                        "symbol": pos.symbol,
                        "quantity": int(pos.qty),
                        "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "current_price": float(pos.current_price),
                        "avg_entry_price": float(pos.avg_entry_price),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account equity and buying power.

        Returns:
            Dictionary with account information
        """
        if not self.trading_client:
            return {
                "equity": 0.0,
                "buying_power": 0.0,
                "cash": 0.0,
            }

        try:
            account = self.trading_client.get_account()

            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }

        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {
                "equity": 0.0,
                "buying_power": 0.0,
                "cash": 0.0,
                "error": str(e),
            }

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        if not self.trading_client:
            return False

        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            List of open order dictionaries
        """
        if not self.trading_client:
            return []

        try:
            orders = self.trading_client.get_orders(status="open")

            result = []
            for order in orders:
                result.append(
                    {
                        "order_id": str(order.id),
                        "symbol": order.symbol,
                        "quantity": int(order.qty),
                        "side": (
                            order.side.value if hasattr(order.side, "value") else str(order.side)
                        ),
                        "order_type": (
                            order.order_type.value
                            if hasattr(order.order_type, "value")
                            else str(order.order_type)
                        ),
                        "status": (
                            order.status.value
                            if hasattr(order.status, "value")
                            else str(order.status)
                        ),
                        "submitted_at": (
                            order.submitted_at.isoformat()
                            if hasattr(order, "submitted_at") and order.submitted_at
                            else None
                        ),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
