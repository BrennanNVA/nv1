"""Automated model training pipeline with walk-forward validation.

Institutional-grade training workflow that:
- Fetches historical data for training universe
- Applies NPMM labeling methodology
- Runs walk-forward optimization with statistical validation
- Saves trained models to registry with versioning
- Generates comprehensive training reports
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import xgboost as xgb

# Initialize logger first (before any try/except blocks that use it)
logger = logging.getLogger(__name__)

# Free alternative to MLFinLab for purged cross-validation
try:
    from ..features.finml_utils import PurgedKFold

    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False
    logger.warning("Purged CV not available - using standard time-series split")

from ..core.config import Config
from ..data.loader import DataLoader
from ..data.storage import StorageService
from ..features.technical import TechnicalFeatures
from .trainer import ModelTrainer
from .validation import BacktestValidator, WalkForwardOptimizer


class TrainingPipeline:
    """Automated model training pipeline with walk-forward validation."""

    def __init__(
        self,
        config: Config,
        data_loader: DataLoader,
        storage: Optional[StorageService] = None,
    ) -> None:
        """
        Initialize training pipeline.

        Args:
            config: System configuration
            data_loader: Data loader for fetching historical data
            storage: Optional storage service for caching data
        """
        self.config = config
        self.data_loader = data_loader
        self.storage = storage

        self.technical_features = TechnicalFeatures(config.technical)
        self.trainer = ModelTrainer(config.ml)
        self.validator = BacktestValidator()
        self.wfo = WalkForwardOptimizer(
            is_window_years=2,
            oos_window_months=6,
            step_months=3,
            expanding=False,
        )

        # Model registry path
        project_root = Path(__file__).parent.parent.parent.parent
        self.model_dir = project_root / "models"
        self.model_dir.mkdir(exist_ok=True)

        logger.info(f"TrainingPipeline initialized (model_dir: {self.model_dir})")

    async def train_universe(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        use_walk_forward: bool = True,
    ) -> dict[str, Any]:
        """
        Train models for a universe of symbols with async pipeline optimization.

        Optimized to overlap CPU preprocessing (data fetching, feature calculation)
        with GPU training for maximum efficiency on RTX 5070 Ti + Ryzen 7700x.

        Args:
            symbols: List of stock symbols to train on
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            use_walk_forward: Whether to use walk-forward optimization

        Returns:
            Dictionary with training results and metrics
        """
        logger.info(
            f"Starting universe training: {len(symbols)} symbols from {start_date} to {end_date}"
        )

        all_results = []
        successful_models = []
        failed_symbols = []

        # Async pipeline: prepare next symbol's data while training current symbol
        import asyncio

        async def prepare_symbol_data(symbol: str) -> Optional[dict[str, Any]]:
            """Prepare data and features for a symbol (CPU-bound preprocessing)."""
            try:
                logger.debug(f"Preparing data for {symbol}...")

                # Check database cache first (much faster than API fetch)
                df = None
                if self.storage:
                    try:
                        logger.debug(f"Checking database cache for {symbol}...")
                        df = await self.storage.load_bars(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        # Check if cached data covers required date range
                        if not df.is_empty():
                            from datetime import datetime as dt

                            # Convert start_date/end_date to datetime if strings
                            req_start = (
                                dt.fromisoformat(
                                    start_date.replace("Z", "+00:00").replace(" ", "T")
                                )
                                if isinstance(start_date, str)
                                else start_date
                            )
                            req_end = (
                                dt.fromisoformat(end_date.replace("Z", "+00:00").replace(" ", "T"))
                                if isinstance(end_date, str)
                                else end_date
                            )

                            data_start = df["timestamp"].min()
                            data_end = df["timestamp"].max()

                            # If cached data covers the required range, use it
                            if data_start <= req_start and data_end >= req_end:
                                logger.info(
                                    f"✓ Using cached data for {symbol} ({len(df)} bars from database)"
                                )
                            else:
                                logger.debug(
                                    f"Cached data incomplete for {symbol} (has {data_start} to {data_end}, need {req_start} to {req_end})"
                                )
                                df = None  # Will fetch from API to get complete range
                        else:
                            logger.debug(f"No cached data found for {symbol} in database")
                    except Exception as e:
                        logger.debug(
                            f"Could not load from cache for {symbol}: {e}, will fetch from API"
                        )
                        df = None

                # If no cache or incomplete, fetch from API
                if df is None or df.is_empty():
                    logger.debug(f"Fetching {symbol} from API...")
                    df = await self.data_loader.fetch_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=self.config.data.default_timeframe,
                    )

                    # Store in database for next time (caching)
                    if self.storage and not df.is_empty():
                        await self.storage.store_bars(df, symbol)

                if df.is_empty():
                    return None

                # Calculate technical features (CPU-bound, but async allows other work)
                # Run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                features_df = await loop.run_in_executor(
                    None,
                    lambda: self.technical_features.calculate_ml_features(
                        df,
                        apply_ffd=True,
                        apply_zscore=True,
                    ),
                )

                if features_df.is_empty():
                    return None

                # Generate NPMM labels (CPU-bound)
                features_df = await loop.run_in_executor(
                    None,
                    lambda: self.trainer.labeler.generate_binary_labels(
                        features_df,
                        price_col="close",
                    ),
                )

                # Filter invalid labels
                features_df = features_df.filter(pl.col("label") >= 0)

                if features_df.is_empty():
                    return None

                return {
                    "symbol": symbol,
                    "features_df": features_df,
                }
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {e}", exc_info=True)
                return None

        # Process symbols with async pipeline optimization
        prepared_data = None

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Training model for {symbol} ({i+1}/{len(symbols)})...")

                # Start preparing next symbol's data while training current (if not last)
                next_prep_task = None
                if i < len(symbols) - 1:
                    next_symbol = symbols[i + 1]
                    next_prep_task = asyncio.create_task(prepare_symbol_data(next_symbol))

                # Use prepared data if available, otherwise prepare now
                if prepared_data and prepared_data["symbol"] == symbol:
                    features_df = prepared_data["features_df"]
                    # Fetch data was already done, but we need to run training
                    if use_walk_forward:
                        result = await self._train_with_prepared_data_wfo(
                            symbol=symbol,
                            features_df=features_df,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    else:
                        result = await self._train_with_prepared_data(
                            symbol=symbol,
                            features_df=features_df,
                        )
                else:
                    # Prepare data now (no overlap)
                    prep_result = await prepare_symbol_data(symbol)
                    if prep_result is None:
                        failed_symbols.append(symbol)
                        logger.warning(f"Data preparation failed for {symbol}")
                        continue

                    features_df = prep_result["features_df"]
                    if use_walk_forward:
                        result = await self._train_with_prepared_data_wfo(
                            symbol=symbol,
                            features_df=features_df,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    else:
                        result = await self._train_with_prepared_data(
                            symbol=symbol,
                            features_df=features_df,
                        )

                # Wait for next symbol's data preparation to complete
                if next_prep_task:
                    prepared_data = await next_prep_task
                    if prepared_data:
                        logger.debug(
                            f"Prepared data for {prepared_data['symbol']} while training {symbol}"
                        )

                if result["success"]:
                    successful_models.append(symbol)
                    all_results.append(result)
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"Training failed for {symbol}: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error training {symbol}: {e}", exc_info=True)
                failed_symbols.append(symbol)

        # Generate summary report
        summary = {
            "total_symbols": len(symbols),
            "successful": len(successful_models),
            "failed": len(failed_symbols),
            "failed_symbols": failed_symbols,
            "training_date": datetime.now().isoformat(),
            "date_range": {"start": start_date, "end": end_date},
            "results": all_results,
        }

        # Save summary report
        report_path = (
            self.model_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Universe training complete: {len(successful_models)}/{len(symbols)} successful. "
            f"Report saved to {report_path}"
        )

        return summary

    async def train_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Train a single model for one symbol.

        Args:
            symbol: Stock symbol
            start_date: Training start date
            end_date: Training end date

        Returns:
            Training result dictionary
        """
        try:
            # Fetch historical data
            logger.debug(f"Fetching data for {symbol}...")
            df = await self.data_loader.fetch_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.data.default_timeframe,
            )

            if df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "No data fetched",
                }

            # Store in database if available
            if self.storage:
                await self.storage.store_bars(df, symbol)

            # Calculate technical features
            logger.debug(f"Calculating features for {symbol}...")
            features_df = self.technical_features.calculate_ml_features(
                df,
                apply_ffd=True,
                apply_zscore=True,
            )

            if features_df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Feature calculation failed",
                }

            # Generate labels using NPMM methodology BEFORE training
            # train_with_npmm generates labels internally, but we need them for validation too
            logger.debug(f"Generating NPMM labels for {symbol}...")
            features_df = self.trainer.labeler.generate_binary_labels(
                features_df,
                price_col="close",
            )

            # Filter out invalid labels
            features_df = features_df.filter(pl.col("label") >= 0)

            if features_df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "No valid labels generated",
                }

            # Get feature names (excluding label)
            feature_names = self.technical_features.get_feature_names()
            feature_names = [f for f in feature_names if f != "label"]

            # Train model with NPMM labeling
            logger.debug(f"Training model for {symbol}...")
            model, metrics = self.trainer.train_with_npmm(
                df=features_df,
                feature_cols=feature_names,
                val_ratio=0.2,
                optimize=True,
            )

            # Validate model
            X = features_df.select(feature_names)
            y = features_df["label"]

            cv_results = self.trainer.cross_validate(
                X=X,
                y=y,
                n_splits=5,
                purge_gap=5,
            )

            # Save model
            model_version = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            model_path = self.model_dir / f"{model_version}.json"
            self.trainer.save_model(str(model_path), include_metadata=True)

            # Calculate validation metrics
            # Use return_1d if available, otherwise calculate from close prices
            if "return_1d" in features_df.columns:
                returns = features_df["return_1d"].to_numpy()
            elif "close" in features_df.columns:
                returns = features_df["close"].pct_change().drop_nulls().to_numpy()
            else:
                returns = np.array([0.0])  # Fallback

            all_trials_sr = [cv_results["mean_score"]]  # Simplified for single model

            dsr = (
                self.validator.calculate_dsr(
                    returns=returns,
                    all_trials_sr=all_trials_sr,
                )
                if len(returns) > 0
                else 0.0
            )

            result = {
                "success": True,
                "symbol": symbol,
                "model_version": model_version,
                "model_path": str(model_path),
                "metrics": metrics,
                "cv_results": cv_results,
                "dsr": dsr,
                "n_samples": len(features_df),
                "n_features": len(feature_names),
            }

            logger.info(
                f"Model trained for {symbol}: Accuracy={metrics['accuracy']:.3f}, "
                f"DSR={dsr:.3f}, CV={cv_results['mean_score']:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error training {symbol}: {e}", exc_info=True)
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
            }

    async def walk_forward_train_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Train model using walk-forward optimization.

        Args:
            symbol: Stock symbol
            start_date: Training start date
            end_date: Training end date

        Returns:
            Walk-forward training results
        """
        try:
            # Fetch full historical data
            df = await self.data_loader.fetch_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=self.config.data.default_timeframe,
            )

            if df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "No data fetched",
                }

            # Calculate features
            features_df = self.technical_features.calculate_ml_features(
                df,
                apply_ffd=True,
                apply_zscore=True,
            )

            if features_df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Feature calculation failed",
                }

            # Generate labels using NPMM methodology
            # This must be done before walk-forward windows are created
            logger.debug(f"Generating NPMM labels for {symbol}...")
            features_df = self.trainer.labeler.generate_binary_labels(
                features_df,
                price_col="close",
            )

            # Filter out invalid labels (last n_period rows have no future data)
            features_df = features_df.filter(pl.col("label") >= 0)

            if features_df.is_empty():
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "No valid labels generated",
                }

            logger.debug(f"Generated labels: {features_df['label'].value_counts()}")

            # Generate walk-forward windows
            windows = self.wfo.generate_windows(features_df)

            if not windows:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Insufficient data for walk-forward windows",
                }

            logger.info(f"Walk-forward training for {symbol}: {len(windows)} windows")

            # Train on each window
            wfo_results = []
            all_oos_returns = []

            feature_names = self.technical_features.get_feature_names()
            # Ensure feature_names don't include 'label' if it's in the list
            feature_names = [f for f in feature_names if f != "label"]

            for i, window in enumerate(windows):
                logger.debug(
                    f"Window {i+1}/{len(windows)}: IS={window['is_start']}-{window['is_end']}, OOS={window['oos_start']}-{window['oos_end']}"
                )

                # Split data
                is_data = features_df[window["is_start"] : window["is_end"]]
                oos_data = features_df[window["oos_start"] : window["oos_end"]]

                if len(is_data) < 100 or len(oos_data) < 20:
                    logger.warning(f"Window {i+1} skipped: insufficient data")
                    continue

                # Train on IS
                X_is = is_data.select(feature_names)
                y_is = is_data["label"]

                model = self.trainer.train(
                    X_train=X_is,
                    y_train=y_is,
                    optimize=True,
                )

                # Validate on OOS
                X_oos = oos_data.select(feature_names)
                y_oos = oos_data["label"]

                y_pred = model.predict(X_oos.to_numpy())
                oos_accuracy = (y_pred == y_oos.to_numpy()).mean()

                # Calculate OOS returns
                if "return_1d" in oos_data.columns:
                    oos_returns = oos_data["return_1d"].to_numpy()
                elif "close" in oos_data.columns:
                    oos_returns = oos_data["close"].pct_change().drop_nulls().to_numpy()
                else:
                    oos_returns = np.array([0.0])

                all_oos_returns.extend(oos_returns.tolist())

                wfo_results.append(
                    {
                        "window": i + 1,
                        "is_start": window["is_start_dt"],
                        "is_end": window["is_end_dt"],
                        "oos_start": window["oos_start_dt"],
                        "oos_end": window["oos_end_dt"],
                        "oos_accuracy": float(oos_accuracy),
                        "oos_sharpe": self.validator.calculate_sharpe_ratio(pl.Series(oos_returns)),
                    }
                )

            # Aggregate OOS performance
            if all_oos_returns:
                aggregated_sharpe = self.validator.calculate_sharpe_ratio(
                    pl.Series(all_oos_returns)
                )
                aggregated_psr = self.validator.calculate_psr(pl.Series(all_oos_returns))
            else:
                aggregated_sharpe = 0.0
                aggregated_psr = 0.0

            # Train final model on all data
            X_all = features_df.select(feature_names)
            y_all = features_df["label"]

            final_model, final_metrics = self.trainer.train_with_npmm(
                df=features_df,
                feature_cols=feature_names,
                val_ratio=0.2,
                optimize=True,
            )

            # Save final model
            model_version = f"{symbol}_wfo_{datetime.now().strftime('%Y%m%d')}"
            model_path = self.model_dir / f"{model_version}.json"
            self.trainer.save_model(str(model_path), include_metadata=True)

            result = {
                "success": True,
                "symbol": symbol,
                "model_version": model_version,
                "model_path": str(model_path),
                "n_windows": len(windows),
                "wfo_results": wfo_results,
                "aggregated_sharpe": aggregated_sharpe,
                "aggregated_psr": aggregated_psr,
                "final_metrics": final_metrics,
            }

            logger.info(
                f"WFO training complete for {symbol}: "
                f"Sharpe={aggregated_sharpe:.3f}, PSR={aggregated_psr:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in WFO training for {symbol}: {e}", exc_info=True)
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
            }

    async def _train_with_prepared_data(
        self,
        symbol: str,
        features_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Train model using pre-prepared features DataFrame.

        This method is optimized for async pipeline where data is already prepared.

        Args:
            symbol: Stock symbol
            features_df: Pre-calculated features DataFrame with labels

        Returns:
            Training result dictionary
        """
        try:
            # Get feature names
            feature_names = self.technical_features.get_feature_names()
            feature_names = [f for f in feature_names if f != "label"]

            # Train model (GPU-bound, run in executor to avoid blocking)
            import asyncio

            loop = asyncio.get_event_loop()
            model, metrics = await loop.run_in_executor(
                None,
                lambda: self.trainer.train_with_npmm(
                    df=features_df,
                    feature_cols=feature_names,
                    val_ratio=0.2,
                    optimize=True,
                ),
            )

            # Validate model
            X = features_df.select(feature_names)
            y = features_df["label"]

            cv_results = await loop.run_in_executor(
                None,
                lambda: self.trainer.cross_validate(
                    X=X,
                    y=y,
                    n_splits=5,
                    purge_gap=5,
                ),
            )

            # Save model
            model_version = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            model_path = self.model_dir / f"{model_version}.json"
            await loop.run_in_executor(
                None, lambda: self.trainer.save_model(str(model_path), include_metadata=True)
            )

            # Calculate validation metrics
            if "return_1d" in features_df.columns:
                returns = features_df["return_1d"].to_numpy()
            elif "close" in features_df.columns:
                returns = features_df["close"].pct_change().drop_nulls().to_numpy()
            else:
                returns = np.array([0.0])

            all_trials_sr = [cv_results["mean_score"]]

            dsr = (
                self.validator.calculate_dsr(
                    returns=returns,
                    all_trials_sr=all_trials_sr,
                )
                if len(returns) > 0
                else 0.0
            )

            result = {
                "success": True,
                "symbol": symbol,
                "model_version": model_version,
                "model_path": str(model_path),
                "metrics": metrics,
                "cv_results": cv_results,
                "dsr": dsr,
                "n_samples": len(features_df),
                "n_features": len(feature_names),
            }

            logger.info(
                f"Model trained for {symbol}: Accuracy={metrics['accuracy']:.3f}, "
                f"DSR={dsr:.3f}, CV={cv_results['mean_score']:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error training {symbol} with prepared data: {e}", exc_info=True)
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
            }

    async def _train_with_prepared_data_wfo(
        self,
        symbol: str,
        features_df: pl.DataFrame,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Train model using walk-forward optimization with pre-prepared features.

        Args:
            symbol: Stock symbol
            features_df: Pre-calculated features DataFrame with labels
            start_date: Training start date
            end_date: Training end date

        Returns:
            Walk-forward training results
        """
        try:
            # Generate walk-forward windows
            windows = self.wfo.generate_windows(features_df)

            if not windows:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Insufficient data for walk-forward windows",
                }

            logger.info(f"Walk-forward training for {symbol}: {len(windows)} windows")

            # Train on each window (GPU-bound, run in executor)
            import asyncio

            loop = asyncio.get_event_loop()

            wfo_results = []
            all_oos_returns = []

            feature_names = self.technical_features.get_feature_names()
            feature_names = [f for f in feature_names if f != "label"]

            for i, window in enumerate(windows):
                logger.debug(
                    f"Window {i+1}/{len(windows)}: IS={window['is_start']}-{window['is_end']}, OOS={window['oos_start']}-{window['oos_end']}"
                )

                # Split data
                is_data = features_df[window["is_start"] : window["is_end"]]
                oos_data = features_df[window["oos_start"] : window["oos_end"]]

                if len(is_data) < 100 or len(oos_data) < 20:
                    logger.warning(f"Window {i+1} skipped: insufficient data")
                    continue

                # Train on IS (GPU-bound)
                X_is = is_data.select(feature_names)
                y_is = is_data["label"]

                model = await loop.run_in_executor(
                    None,
                    lambda: self.trainer.train(
                        X_train=X_is,
                        y_train=y_is,
                        optimize=True,
                    ),
                )

                # Validate on OOS
                X_oos = oos_data.select(feature_names)
                y_oos = oos_data["label"]

                y_pred = model.predict(X_oos.to_numpy())
                oos_accuracy = (y_pred == y_oos.to_numpy()).mean()

                # Calculate OOS returns
                if "return_1d" in oos_data.columns:
                    oos_returns = oos_data["return_1d"].to_numpy()
                elif "close" in oos_data.columns:
                    oos_returns = oos_data["close"].pct_change().drop_nulls().to_numpy()
                else:
                    oos_returns = np.array([0.0])

                all_oos_returns.extend(oos_returns.tolist())

                wfo_results.append(
                    {
                        "window": i + 1,
                        "is_start": window["is_start_dt"],
                        "is_end": window["is_end_dt"],
                        "oos_start": window["oos_start_dt"],
                        "oos_end": window["oos_end_dt"],
                        "oos_accuracy": float(oos_accuracy),
                        "oos_sharpe": self.validator.calculate_sharpe_ratio(pl.Series(oos_returns)),
                    }
                )

            # Aggregate OOS performance
            if all_oos_returns:
                aggregated_sharpe = self.validator.calculate_sharpe_ratio(
                    pl.Series(all_oos_returns)
                )
                aggregated_psr = self.validator.calculate_psr(pl.Series(all_oos_returns))
            else:
                aggregated_sharpe = 0.0
                aggregated_psr = 0.0

            # Train final model on all data (GPU-bound)
            final_model, final_metrics = await loop.run_in_executor(
                None,
                lambda: self.trainer.train_with_npmm(
                    df=features_df,
                    feature_cols=feature_names,
                    val_ratio=0.2,
                    optimize=True,
                ),
            )

            # Save final model
            model_version = f"{symbol}_wfo_{datetime.now().strftime('%Y%m%d')}"
            model_path = self.model_dir / f"{model_version}.json"
            await loop.run_in_executor(
                None, lambda: self.trainer.save_model(str(model_path), include_metadata=True)
            )

            result = {
                "success": True,
                "symbol": symbol,
                "model_version": model_version,
                "model_path": str(model_path),
                "n_windows": len(windows),
                "wfo_results": wfo_results,
                "aggregated_sharpe": aggregated_sharpe,
                "aggregated_psr": aggregated_psr,
                "final_metrics": final_metrics,
            }

            logger.info(
                f"WFO training complete for {symbol}: "
                f"Sharpe={aggregated_sharpe:.3f}, PSR={aggregated_psr:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in WFO training for {symbol}: {e}", exc_info=True)
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
            }

    def save_model_registry(
        self,
        model: xgb.XGBClassifier,
        metrics: dict[str, Any],
        version: str,
    ) -> Path:
        """
        Save model to registry with metadata.

        Args:
            model: Trained XGBoost model
            metrics: Training metrics
            version: Model version string

        Returns:
            Path to saved model file
        """
        model_path = self.model_dir / f"{version}.json"
        metadata_path = self.model_dir / f"{version}_metadata.json"

        # Save model
        model.save_model(str(model_path))

        # Save metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "config": {
                "n_estimators": self.config.ml.n_estimators,
                "max_depth": self.config.ml.max_depth,
                "learning_rate": self.config.ml.learning_rate,
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to registry: {model_path}")
        return model_path
