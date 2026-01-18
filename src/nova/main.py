"""Main entry point for Nova Aetus trading system."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Load environment variables first
from dotenv import load_dotenv

# Load .env from current directory and parent directory
load_dotenv()  # Current directory
parent_env = Path(__file__).parent.parent.parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env, override=False)  # Parent directory (don't override current dir)

import numpy as np

from .api.health_server import HealthServer
from .core.config import Config, load_config
from .core.health import HealthChecker
from .core.logger import CircuitBreaker, setup_logger
from .core.metrics import get_metrics
from .core.notifications import NotificationService
from .data.loader import DataLoader
from .data.storage import StorageService
from .features.sentiment import SentimentAnalyzer
from .features.technical import TechnicalFeatures
from .models.predictor import ModelRegistry
from .strategy.confluence import ConfluenceLayer
from .strategy.execution import ExecutionEngine
from .strategy.portfolio_optimizer import OptimizationMethod, PortfolioOptimizer
from .strategy.risk import RiskManager

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# Global shutdown flag
_shutdown_event: Optional[asyncio.Event] = None


async def trading_loop(
    data_loader: DataLoader,
    storage: Optional[StorageService],
    technical_features: TechnicalFeatures,
    sentiment_analyzer: SentimentAnalyzer,
    confluence_layer: ConfluenceLayer,
    risk_manager: RiskManager,
    execution_engine: ExecutionEngine,
    model_registry: Optional[ModelRegistry],
    portfolio_optimizer: Optional[PortfolioOptimizer],
    config: Config,
    metrics: Any,
    notifications: NotificationService,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Main trading loop with complete signal generation cycle.

    Implements institutional-grade trading workflow:
    1. Fetch latest data for all symbols
    2. Calculate technical indicators
    3. Get sentiment scores (news/social)
    4. Generate confluence signals
    5. Evaluate trades with risk manager
    6. Execute approved trades
    7. Update positions and P&L
    8. Store signals and metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Trading loop started")

    # Initialize risk manager with starting equity
    # This would come from account info in production
    starting_equity = 100000.0  # Default, should come from Alpaca account
    risk_manager.set_starting_equity(starting_equity)
    risk_manager.reset_daily_tracking()

    # Portfolio rebalancing state (swing trading: rebalance once per day or weekly)
    last_rebalance_date: Optional[datetime] = None
    use_portfolio_optimization = portfolio_optimizer is not None

    if use_portfolio_optimization:
        logger.info("Portfolio optimization enabled - collecting signals and optimizing portfolio")
    else:
        logger.info("Portfolio optimization disabled - using individual trade execution")

    # Main loop - runs until shutdown
    iteration = 0
    while not shutdown_event.is_set():
        try:
            iteration += 1
            logger.info(f"Trading loop iteration {iteration}")

            # Check circuit breaker
            can_trade, reason = risk_manager.check_can_trade()
            if not can_trade:
                logger.warning(f"Trading halted: {reason}")
                await notifications.send_alert(
                    "CIRCUIT_BREAKER",
                    f"Trading halted: {reason}",
                )
                # Wait before retrying
                await asyncio.sleep(60)
                continue

            # Reset daily tracking if new day
            risk_manager.reset_daily_tracking()

            # Check if we should rebalance portfolio (swing trading: once per day)
            should_rebalance = False
            if use_portfolio_optimization:
                current_date = datetime.now().date()
                if last_rebalance_date is None or last_rebalance_date.date() < current_date:
                    should_rebalance = True
                    logger.info("Portfolio rebalancing scheduled (once per day for swing trading)")

            # Collect all signals first (portfolio approach) or process individually
            all_signals: dict[str, Any] = {}  # symbol -> {signal, price, stop_loss, etc.}

            # Step 1: Collect individual predictions from all symbols for master model
            individual_predictions: dict[
                str, dict[str, float]
            ] = {}  # symbol -> {score, confidence, prediction}
            symbol_data_cache: dict[
                str, dict[str, Any]
            ] = {}  # symbol -> {df, features_df, market_data, technical_signals}

            # Get master model if available
            master_model = None
            if model_registry:
                master_model = model_registry.get_master_model()

            # First pass: Collect all individual predictions
            for symbol in config.data.symbols:
                if shutdown_event.is_set():
                    break

                try:
                    # 1. Fetch latest data
                    logger.debug(f"Fetching data for {symbol}...")
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (
                        datetime.now() - timedelta(days=config.data.lookback_periods)
                    ).strftime("%Y-%m-%d")

                    df = await data_loader.fetch_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=config.data.default_timeframe,
                    )

                    if df.is_empty():
                        logger.warning(f"No data for {symbol}")
                        continue

                    # Store in database if available
                    if storage:
                        await storage.store_bars(df, symbol)

                    # 2. Calculate technical indicators
                    logger.debug(f"Calculating technical features for {symbol}...")
                    features_df = technical_features.calculate_ml_features(
                        df,
                        apply_ffd=True,
                        apply_zscore=True,
                    )

                    if features_df.is_empty():
                        logger.warning(f"No features generated for {symbol}")
                        continue

                    # Get latest row for prediction
                    latest_features = features_df.tail(1)

                    # 3. Get technical prediction from individual model
                    technical_score = 0.0
                    technical_signals = {}

                    # Load symbol-specific model from registry
                    predictor = None
                    if model_registry:
                        predictor = model_registry.get_predictor(symbol)

                    if predictor:
                        try:
                            feature_names = technical_features.get_feature_names()
                            feature_cols = [
                                col for col in feature_names if col in latest_features.columns
                            ]

                            if feature_cols:
                                pred_df = predictor.predict_with_confidence(
                                    latest_features.select(feature_cols)
                                )
                                raw_score = (
                                    float(pred_df["prediction"][0]) * 2 - 1
                                )  # Convert 0/1 to -1/1
                                confidence = float(pred_df["confidence"][0])

                                # Store individual prediction for master model
                                individual_predictions[symbol] = {
                                    "score": raw_score,
                                    "confidence": confidence,
                                    "prediction": float(pred_df["prediction"][0]),
                                }

                                # Extract individual technical signals
                                tech_signals = {}
                                for col in [
                                    "rsi",
                                    "macd",
                                    "macd_histogram",
                                    "bb_pct_b",
                                    "adx",
                                    "stoch_k",
                                ]:
                                    if col in latest_features.columns:
                                        tech_signals[col] = float(latest_features[col][0])

                                # Store market data for master model
                                volatility = (
                                    float(df["close"].std() / df["close"].mean())
                                    if len(df) > 0
                                    else 0.015
                                )
                                trend_strength = (
                                    float(latest_features["adx"][0])
                                    if "adx" in latest_features.columns
                                    else 20.0
                                )
                                returns = (
                                    df["close"].pct_change().drop_nulls().to_numpy()[-20:]
                                    if len(df) >= 20
                                    else np.array([0])
                                )

                                # Cache all data for this symbol
                                symbol_data_cache[symbol] = {
                                    "df": df,
                                    "features_df": latest_features,
                                    "technical_signals": tech_signals,
                                    "market_data": {
                                        "volatility": volatility,
                                        "trend_strength": trend_strength,
                                        "returns": returns.tolist(),
                                    },
                                }

                                technical_score = (
                                    raw_score  # Will be improved by master model if available
                                )
                        except Exception as e:
                            logger.error(f"Technical prediction failed for {symbol}: {e}")
                            metrics.record_error("model", "prediction_failed")

                    # Store other data we'll need later
                    if symbol not in symbol_data_cache:
                        symbol_data_cache[symbol] = {
                            "df": df,
                            "features_df": latest_features,
                            "technical_signals": technical_signals,
                            "market_data": {
                                "volatility": (
                                    float(df["close"].std() / df["close"].mean())
                                    if len(df) > 0
                                    else 0.015
                                ),
                                "trend_strength": (
                                    float(latest_features["adx"][0])
                                    if "adx" in latest_features.columns
                                    else 20.0
                                ),
                                "returns": (
                                    df["close"].pct_change().drop_nulls().to_numpy()[-20:].tolist()
                                    if len(df) >= 20
                                    else [0]
                                ),
                            },
                        }
                except Exception as e:
                    logger.error(f"Failed to process {symbol} in first pass: {e}")
                    metrics.record_error("data", "symbol_processing_failed")

            # Step 2: Apply master ensemble model to improve predictions
            if master_model and individual_predictions:
                logger.debug("Applying master ensemble model to improve predictions...")
                # Calculate aggregate market data (average across all symbols)
                market_data_list = [
                    cache["market_data"]
                    for cache in symbol_data_cache.values()
                    if "market_data" in cache
                ]
                if market_data_list:
                    avg_volatility = np.mean([d["volatility"] for d in market_data_list])
                    avg_trend = np.mean([d["trend_strength"] for d in market_data_list])
                    all_returns = []
                    for d in market_data_list:
                        all_returns.extend(d.get("returns", []))
                    aggregate_market_data = {
                        "volatility": avg_volatility,
                        "trend_strength": avg_trend,
                        "returns": all_returns[-20:] if len(all_returns) >= 20 else all_returns,
                    }
                else:
                    aggregate_market_data = None

                # Improve each symbol's prediction using master model
                for symbol in individual_predictions.keys():
                    try:
                        improved = master_model.predict(
                            individual_predictions, aggregate_market_data, symbol
                        )
                        # Use improved score if master model provides it
                        if "improved_score" in improved:
                            individual_predictions[symbol]["score"] = improved["improved_score"]
                            logger.debug(
                                f"Master model improved {symbol}: "
                                f"{improved.get('original_score', 0):.3f} -> "
                                f"{improved['improved_score']:.3f}"
                            )
                    except Exception as e:
                        logger.warning(f"Master model prediction failed for {symbol}: {e}")

            # Step 3: Continue with confluence layer using improved predictions
            for symbol in config.data.symbols:
                if shutdown_event.is_set():
                    break

                try:
                    # Skip if we don't have cached data for this symbol
                    if symbol not in symbol_data_cache:
                        continue

                    # Get cached data
                    cached = symbol_data_cache[symbol]
                    df = cached["df"]
                    latest_features = cached["features_df"]
                    technical_signals = cached["technical_signals"]
                    market_data_dict = cached["market_data"]

                    # Get improved technical score from master model (or original if no master)
                    if symbol in individual_predictions:
                        technical_score = individual_predictions[symbol]["score"]
                    else:
                        technical_score = 0.0

                    # 4. Get sentiment analysis
                    sentiment_score = 0.0
                    sentiment_data: dict[str, Any] = {}

                    # In production, fetch news/social data here
                    # For now, use placeholder sentiment
                    try:
                        # TODO: Fetch actual news/social data
                        # news_items = await fetch_news_for_symbol(symbol)
                        # sentiment_result = await sentiment_analyzer.get_aggregated_sentiment(news_items, symbol)
                        # sentiment_score = sentiment_result["aggregated_score"]
                        # sentiment_data = sentiment_result
                        pass
                    except Exception as e:
                        logger.debug(f"Sentiment analysis skipped for {symbol}: {e}")

                    # 5. Get fundamental data
                    fundamental_data = None
                    try:
                        fundamental_result = await data_loader.fetch_fundamentals(symbol)
                        if fundamental_result:
                            # Extract scores for confluence layer
                            fundamental_data = {
                                "value_score": fundamental_result.get("value_score", 0.0),
                                "quality_score": fundamental_result.get("quality_score", 0.0),
                                "pe_ratio": fundamental_result.get("pe_ratio"),
                                "pb_ratio": fundamental_result.get("pb_ratio"),
                                "roe": fundamental_result.get("roe"),
                            }
                    except Exception as e:
                        logger.debug(f"Fundamental data fetch failed for {symbol}: {e}")

                    # 6. Generate confluence signal
                    try:
                        # Prepare market data for regime detection
                        returns = (
                            df["close"].pct_change().drop_nulls().to_numpy()[-20:]
                            if len(df) >= 20
                            else np.array([0])
                        )
                        volatility = (
                            float(df["close"].std() / df["close"].mean()) if len(df) > 0 else 0.015
                        )
                        trend_strength = (
                            float(latest_features["adx"][0])
                            if "adx" in latest_features.columns
                            else 20.0
                        )

                        market_data = {
                            "returns": returns.tolist(),
                            "volatility": volatility,
                            "adx": trend_strength,
                        }

                        confluence_signal = confluence_layer.generate_confluence_signal(
                            technical_signals=technical_signals,
                            sentiment_data={"score": sentiment_score, **sentiment_data},
                            fundamental_data=fundamental_data,
                            market_data=market_data,
                        )

                        # 7. Store signal in database
                        if storage:
                            await storage.store_signal(
                                symbol=symbol,
                                timestamp=datetime.utcnow(),
                                signal_type=(
                                    "ENTRY"
                                    if confluence_signal.direction.value != "neutral"
                                    else "HOLD"
                                ),
                                direction=confluence_signal.direction.value.upper(),
                                strength=confluence_signal.strength,
                                technical_score=confluence_signal.technical_score,
                                fundamental_score=confluence_signal.fundamental_score,
                                sentiment_score=confluence_signal.sentiment_score,
                                confluence_score=confluence_signal.confluence_score,
                                model_version="latest",
                                metadata=confluence_signal.to_dict(),
                            )

                        # 8. Store signal data for portfolio optimization or individual execution
                        current_price = float(df["close"][-1])
                        atr = (
                            float(latest_features["atr"][0])
                            if "atr" in latest_features.columns
                            else current_price * 0.02
                        )
                        stop_loss = risk_manager.calculate_atr_stop_loss(
                            entry_price=current_price,
                            atr_value=atr,
                            is_long=(confluence_signal.direction.value == "long"),
                        )
                        take_profit = (
                            current_price + (current_price - stop_loss) * 2
                            if confluence_signal.direction.value == "long"
                            else current_price - (stop_loss - current_price) * 2
                        )

                        # Store for portfolio optimization
                        if confluence_signal.direction.value != "neutral":
                            all_signals[symbol] = {
                                "signal": confluence_signal,
                                "price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "atr": atr,
                                "features_df": features_df,
                            }

                    except Exception as e:
                        logger.error(
                            f"Confluence signal generation failed for {symbol}: {e}", exc_info=True
                        )
                        metrics.record_error("confluence", "signal_generation_failed")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                    risk_manager.record_error("trading_loop")
                    metrics.record_error("trading_loop", "symbol_processing")

            # Portfolio optimization or individual execution
            if use_portfolio_optimization and should_rebalance and all_signals:
                # Portfolio approach: optimize and rebalance
                await _rebalance_portfolio(
                    all_signals=all_signals,
                    portfolio_optimizer=portfolio_optimizer,
                    execution_engine=execution_engine,
                    risk_manager=risk_manager,
                    starting_equity=starting_equity,
                    storage=storage,
                    metrics=metrics,
                    notifications=notifications,
                    logger=logger,
                )
                last_rebalance_date = datetime.now()
            elif not use_portfolio_optimization:
                # Individual trade approach (original behavior)
                for symbol, signal_data in all_signals.items():
                    confluence_signal = signal_data["signal"]
                    current_price = signal_data["price"]
                    stop_loss = signal_data["stop_loss"]
                    take_profit = signal_data["take_profit"]

                    if confluence_signal.strength > 0.5:
                        current_equity = starting_equity
                        trade_eval = risk_manager.evaluate_trade(
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            capital=current_equity,
                            win_probability=confluence_signal.confidence,
                            current_equity=current_equity,
                        )

                        if trade_eval["approved"]:
                            order_side = (
                                "buy" if confluence_signal.direction.value == "long" else "sell"
                            )
                            order_result = await execution_engine.execute_order(
                                symbol=symbol,
                                quantity=trade_eval["position_size"],
                                side=order_side,
                                order_type="market",
                            )

                            if order_result.get("success") and storage:
                                await storage.open_position(
                                    symbol=symbol,
                                    quantity=trade_eval["position_size"],
                                    entry_price=current_price,
                                    position_type=confluence_signal.direction.value.upper(),
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                )

            # Wait before next iteration (e.g., 5 minutes for intraday, or daily for swing)
            logger.info(f"Trading loop iteration {iteration} complete, waiting...")
            await asyncio.sleep(300)  # 5 minutes between iterations

        except Exception as e:
            logger.error(f"Error in trading loop iteration {iteration}: {e}", exc_info=True)
            risk_manager.record_error("trading_loop")
            metrics.record_error("trading_loop", "iteration_error")
            await asyncio.sleep(60)  # Wait before retrying

    logger.info("Trading loop stopped")


async def _rebalance_portfolio(
    all_signals: dict[str, dict],
    portfolio_optimizer: PortfolioOptimizer,
    execution_engine: ExecutionEngine,
    risk_manager: RiskManager,
    starting_equity: float,
    storage: Optional[StorageService],
    metrics: Any,
    notifications: NotificationService,
    logger: logging.Logger,
) -> None:
    """
    Rebalance portfolio using portfolio optimization.

    Collects all signals, optimizes portfolio weights, and executes rebalancing trades.
    """
    try:
        # Extract signal strengths for optimization
        signal_strengths = {}
        signal_data_map = {}

        for symbol, data in all_signals.items():
            signal = data["signal"]
            # Convert confluence score to signal strength for optimization
            # Positive signal = buy, negative = avoid/sell
            signal_strength = (
                signal.confluence_score
                if signal.direction.value == "long"
                else -abs(signal.confluence_score)
            )
            signal_strengths[symbol] = signal_strength
            signal_data_map[symbol] = data

        # Filter to only positive signals (long positions for swing trading)
        positive_signals = {k: v for k, v in signal_strengths.items() if v > 0.1}

        if not positive_signals:
            logger.info("No positive signals for portfolio optimization")
            return

        # Optimize portfolio
        logger.info(f"Optimizing portfolio for {len(positive_signals)} symbols...")
        portfolio_weights = portfolio_optimizer.optimize(signals=positive_signals)

        logger.info(
            f"Portfolio optimization complete: "
            f"expected_return={portfolio_weights.expected_return:.4f}, "
            f"volatility={portfolio_weights.portfolio_volatility:.4f}, "
            f"sharpe={portfolio_weights.sharpe_ratio:.4f}"
        )

        # Get current positions
        current_positions = {}
        if storage:
            try:
                positions_df = await storage.get_open_positions()
                if not positions_df.is_empty():
                    for row in positions_df.to_dicts():
                        symbol = row.get("symbol")
                        quantity = row.get("quantity", 0)
                        if symbol and quantity:
                            current_positions[symbol] = float(quantity)
            except Exception as e:
                logger.debug(f"Could not fetch current positions: {e}")

        # Calculate target positions and execute rebalancing
        total_equity = starting_equity  # TODO: Get from account
        rebalance_executed = False

        for symbol, target_weight in portfolio_weights.weights.items():
            if target_weight < 0.01:  # Skip positions < 1%
                continue

            signal_data = signal_data_map.get(symbol)
            if not signal_data:
                continue

            current_price = signal_data["price"]
            target_value = total_equity * target_weight
            target_shares = int(target_value / current_price)
            current_shares = int(current_positions.get(symbol, 0))

            # Calculate shares to trade
            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) < 1:  # Skip small adjustments
                continue

            # Execute rebalancing trade
            order_side = "buy" if shares_to_trade > 0 else "sell"
            quantity = abs(shares_to_trade)

            logger.info(
                f"Rebalancing {symbol}: {order_side.upper()} {quantity} shares "
                f"(target: {target_shares}, current: {current_shares}, weight: {target_weight:.2%})"
            )

            try:
                order_result = await execution_engine.execute_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=order_side,
                    order_type="market",
                    current_price=current_price,
                )

                if order_result.get("success"):
                    signal = signal_data["signal"]
                    if storage:
                        if shares_to_trade > 0:
                            # Opening new position or adding
                            await storage.open_position(
                                symbol=symbol,
                                quantity=quantity,
                                entry_price=current_price,
                                position_type=signal.direction.value.upper(),
                                stop_loss=signal_data["stop_loss"],
                                take_profit=signal_data["take_profit"],
                            )
                        # Note: Closing positions would be handled separately

                    metrics.increment("portfolio_rebalanced", labels={"symbol": symbol})
                    rebalance_executed = True
                else:
                    logger.error(
                        f"Rebalancing order failed for {symbol}: {order_result.get('error')}"
                    )

            except Exception as e:
                logger.error(f"Error rebalancing {symbol}: {e}", exc_info=True)
                metrics.record_error("portfolio", "rebalance_failed")

        if rebalance_executed:
            await notifications.send_message(
                f"Portfolio rebalanced: {len(portfolio_weights.weights)} positions, "
                f"Sharpe={portfolio_weights.sharpe_ratio:.2f}",
                title="Portfolio Rebalanced",
                level="INFO",
            )
            logger.info("Portfolio rebalancing complete")

    except Exception as e:
        logger.error(f"Portfolio rebalancing error: {e}", exc_info=True)
        metrics.record_error("portfolio", "rebalance_exception")


def _setup_signal_handlers(shutdown_event: asyncio.Event, logger: logging.Logger) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        signame = signal.Signals(signum).name
        logger.info(f"Received {signame}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register handlers for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def main() -> None:
    """Main async event loop orchestration."""
    global _shutdown_event

    # Create shutdown event
    _shutdown_event = asyncio.Event()

    # Load configuration
    config = load_config()

    # Setup logger and circuit breaker
    circuit_breaker = CircuitBreaker(
        max_errors=config.circuit_breaker.max_errors_per_minute,
        time_window=config.circuit_breaker.error_window_seconds,
        halt_on_breach=config.circuit_breaker.halt_on_breach,
    )

    logger = setup_logger(
        name="nova_aetus",
        log_dir=project_root / "logs",
        level="INFO",
        circuit_breaker=circuit_breaker,
    )

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers(_shutdown_event, logger)

    logger.info("=" * 60)
    logger.info("Nova Aetus Trading System Starting")
    logger.info("=" * 60)

    # Initialize notification service
    notifications = NotificationService(config.notifications)
    await notifications.send_message(
        "Nova Aetus system started",
        title="System Startup",
        level="INFO",
    )

    # Initialize metrics and health
    metrics = get_metrics()
    health_checker = HealthChecker()
    health_server = HealthServer(
        health_checker=health_checker,
        host=config.dashboard.host,
        port=8000,  # Metrics port for Prometheus
    )
    await health_server.start()

    # Initialize services
    storage = None
    try:
        storage = StorageService(config.data)
        try:
            await storage.connect()
            await storage.init_schema()
        except Exception as db_error:
            logger.warning(
                f"Database connection failed (system can still run without DB): {db_error}"
            )
            # Continue without database - system can still function

        data_loader = DataLoader(config.data)
        technical_features = TechnicalFeatures(config.technical)
        sentiment_analyzer = SentimentAnalyzer(config.sentiment)
        risk_manager = RiskManager(config.risk)
        confluence_layer = ConfluenceLayer()

        # Initialize portfolio optimizer (optional - can be None to use individual trades)
        portfolio_optimizer = PortfolioOptimizer(
            method=OptimizationMethod.MEAN_VARIANCE,
            risk_aversion=1.0,
            max_position_weight=config.risk.max_position_size_pct,  # Use risk config
            long_only=True,
        )

        # Initialize execution engine with Alpaca trading client
        execution_engine = ExecutionEngine(
            sentiment_analyzer=sentiment_analyzer,
            data_config=config.data,
        )

        # Initialize model registry for symbol-specific model loading
        # Institutional approach: Individual models per symbol for better accuracy
        model_registry = None
        model_path = project_root / "models"
        if model_path.exists():
            try:
                model_registry = ModelRegistry(model_path)
                available_symbols = model_registry.get_available_symbols()
                if available_symbols:
                    logger.info(
                        f"Model registry initialized: {len(available_symbols)} symbols have trained models"
                    )
                    logger.debug(f"Available models: {', '.join(available_symbols)}")
                else:
                    logger.warning("Model registry initialized but no trained models found")
            except Exception as e:
                logger.warning(f"Could not initialize model registry: {e}")
                model_registry = None

        if model_registry is None:
            logger.warning("No model registry available - signal generation will be limited")

        logger.info("All services initialized")

        # Perform initial health check
        try:
            system_health = await health_checker.get_system_health(
                storage_service=storage if storage and storage.pool else None,
                sentiment_analyzer=sentiment_analyzer,
            )
            logger.info(f"System health: {system_health.status.value}")
            if system_health.any_unhealthy:
                logger.warning("Some components are unhealthy - system running in degraded mode")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")

        # Main trading loop
        logger.info("Starting main trading loop...")

        # Check for shutdown signal
        if _shutdown_event.is_set():
            logger.info("Shutdown requested before starting trading loop")
        else:
            # Start trading loop
            await trading_loop(
                data_loader=data_loader,
                storage=storage,
                technical_features=technical_features,
                sentiment_analyzer=sentiment_analyzer,
                confluence_layer=confluence_layer,
                risk_manager=risk_manager,
                execution_engine=execution_engine,
                model_registry=model_registry,
                portfolio_optimizer=portfolio_optimizer,
                config=config,
                metrics=metrics,
                notifications=notifications,
                shutdown_event=_shutdown_event,
            )

    except SystemExit as e:
        # Circuit breaker halted system
        logger.critical(f"System halted by circuit breaker: {e}")
        await notifications.send_alert(
            "CIRCUIT_BREAKER",
            f"System halted: {e}",
        )
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        circuit_breaker.record_error()
        await notifications.send_alert(
            "ERROR",
            f"Fatal error occurred: {e}",
            {"error_type": type(e).__name__},
        )
        raise
    finally:
        # Graceful cleanup
        logger.info("Initiating graceful shutdown...")

        # Stop health server
        if "health_server" in locals() and health_server:
            await health_server.stop()

        # Close database connections
        if storage is not None:
            try:
                await storage.disconnect()
                logger.info("Database connections closed")
            except Exception as e:
                logger.warning(f"Error closing database connections: {e}")

        # Send shutdown notification
        try:
            await notifications.send_message(
                "Nova Aetus system shutting down gracefully",
                title="System Shutdown",
                level="INFO",
            )
        except Exception as e:
            logger.warning(f"Could not send shutdown notification: {e}")

        logger.info("System shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user (KeyboardInterrupt)")
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 1)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
