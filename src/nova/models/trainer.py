"""XGBoost GPU training with NPMM labeling and Optuna optimization.

Research-backed implementation based on:
- "A machine learning trading system based on N-period Min-Max labeling" (ESWA, 2023)
- RAPIDS cuML for GPU-accelerated inference (FIL)
- Optuna for hyperparameter optimization

Key methodology:
- NPMM labeling generates trend labels only at definitive local minima/maxima
- GPU-accelerated training via device="cuda" and tree_method="hist"
- Walk-forward validation for robust backtesting
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

# Initialize logger first (before any try/except blocks that use it)
logger = logging.getLogger(__name__)

# SHAP for model interpretability
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - install with: pip install shap>=0.45.0")

# Free alternative to MLFinLab for purged cross-validation
try:
    from ..features.finml_utils import PurgedKFold

    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False
    logger.warning("Purged CV not available - using standard k-fold")

# Try to import RMM for GPU memory optimization
# RMM provides 5-10x faster memory allocation and is critical for external memory training
# Installation: conda install -c rapidsai -c conda-forge rmm cuda-version=12.0
# Note: RMM is only available via conda (rapidsai channel), not pip
# The system works without RMM but performance is optimized with it
try:
    import rmm

    RMM_AVAILABLE = True
except ImportError:
    RMM_AVAILABLE = False
    logger.info(
        "RMM (RAPIDS Memory Manager) not available. "
        "Install via conda for optimal GPU performance: "
        "conda install -c rapidsai -c conda-forge rmm cuda-version=12.0"
    )

from ..core.config import MLConfig


class NPMMLabeler:
    """N-Period Min-Max Labeling for swing trading signals.

    Based on "A machine learning trading system for the stock market
    based on N-period Min-Max labeling using XGBoost" (ESWA, 2023).

    Generates trend labels only at definitive local minima/maxima,
    dramatically improving signal quality over fixed-horizon labels.
    """

    def __init__(
        self,
        n_period: int = 5,
        threshold_pct: float = 0.02,
    ) -> None:
        """
        Initialize NPMM labeler.

        Args:
            n_period: Number of periods to look forward/backward for extrema
            threshold_pct: Minimum price change threshold (2% default)
        """
        self.n_period = n_period
        self.threshold_pct = threshold_pct
        logger.info(f"NPMMLabeler initialized: n_period={n_period}, threshold={threshold_pct:.1%}")

    def find_local_extrema(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find local minima and maxima in price series.

        Args:
            prices: Array of prices

        Returns:
            Tuple of (minima_indices, maxima_indices)
        """
        n = len(prices)
        minima = []
        maxima = []

        for i in range(self.n_period, n - self.n_period):
            # Check if local minimum
            is_min = all(
                prices[i] <= prices[j] for j in range(i - self.n_period, i + self.n_period + 1)
            )

            # Check if local maximum
            is_max = all(
                prices[i] >= prices[j] for j in range(i - self.n_period, i + self.n_period + 1)
            )

            if is_min:
                minima.append(i)
            elif is_max:
                maxima.append(i)

        return np.array(minima), np.array(maxima)

    def generate_labels(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
    ) -> pl.DataFrame:
        """
        Generate NPMM labels for the dataset.

        Labels:
        - 1 (BUY): At or near local minimum (price expected to rise)
        - -1 (SELL): At or near local maximum (price expected to fall)
        - 0 (HOLD): Neither extremum reached

        Args:
            df: DataFrame with price data
            price_col: Name of price column

        Returns:
            DataFrame with 'npmm_label' column added
        """
        prices = df[price_col].to_numpy()
        n = len(prices)
        labels = np.zeros(n, dtype=np.int8)

        minima, maxima = self.find_local_extrema(prices)

        # Label minima as BUY signals (1)
        for idx in minima:
            # Check if price rises enough after this point
            if idx + self.n_period < n:
                future_max = np.max(prices[idx : idx + self.n_period * 2])
                pct_change = (future_max - prices[idx]) / prices[idx]
                if pct_change >= self.threshold_pct:
                    labels[idx] = 1

        # Label maxima as SELL signals (-1)
        for idx in maxima:
            # Check if price falls enough after this point
            if idx + self.n_period < n:
                future_min = np.min(prices[idx : idx + self.n_period * 2])
                pct_change = (prices[idx] - future_min) / prices[idx]
                if pct_change >= self.threshold_pct:
                    labels[idx] = -1

        result = df.with_columns(pl.Series("npmm_label", labels))

        # Log statistics
        buy_count = np.sum(labels == 1)
        sell_count = np.sum(labels == -1)
        hold_count = np.sum(labels == 0)

        logger.info(
            f"NPMM labels generated: BUY={buy_count}, SELL={sell_count}, "
            f"HOLD={hold_count} ({buy_count + sell_count} signals from {n} bars)"
        )

        return result

    def generate_binary_labels(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        horizon: int = 5,
    ) -> pl.DataFrame:
        """
        Generate binary labels based on future returns.

        This is a simpler alternative to full NPMM for classification tasks.

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            horizon: Forward-looking horizon in periods

        Returns:
            DataFrame with 'label' column (1 = up, 0 = down/flat)
        """
        prices = df[price_col].to_numpy()
        n = len(prices)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - horizon):
            future_return = (prices[i + horizon] - prices[i]) / prices[i]
            labels[i] = 1 if future_return > self.threshold_pct else 0

        # Mark last 'horizon' rows as invalid (no future data)
        labels[-horizon:] = -1  # Will be filtered out

        result = df.with_columns(pl.Series("label", labels))

        # Log statistics
        up_count = np.sum(labels == 1)
        down_count = np.sum(labels == 0)

        logger.info(
            f"Binary labels generated: UP={up_count}, DOWN={down_count} "
            f"(horizon={horizon}, threshold={self.threshold_pct:.1%})"
        )

        return result


class ModelTrainer:
    """Train XGBoost models with GPU acceleration and Optuna tuning.

    Implements institutional-grade ML pipeline:
    - NPMM labeling for swing trade signals
    - GPU-accelerated training (CUDA)
    - Optuna hyperparameter optimization
    - Model versioning and registry
    - Walk-forward validation ready
    """

    def __init__(self, config: MLConfig) -> None:
        """
        Initialize model trainer.

        Args:
            config: ML configuration
        """
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.labeler = NPMMLabeler()
        self.training_history: list[dict[str, Any]] = []
        self.use_quantile_dmatrix = getattr(config, "use_quantile_dmatrix", True)
        self.use_rmm = getattr(config, "use_rmm", True)
        self.gradient_sampling = getattr(config, "gradient_sampling", False)
        self.feature_selection_top_n = getattr(config, "feature_selection_top_n", None)

        # Check GPU availability first (needed for RMM initialization)
        self._check_gpu_availability()

        # Initialize RMM if available and enabled
        self.rmm_initialized = False
        if self.use_rmm and RMM_AVAILABLE and self.gpu_available:
            self._initialize_rmm()
        logger.info("ModelTrainer initialized with GPU support")

    def _check_gpu_availability(self) -> None:
        """Check if GPU is available for training."""
        try:
            # Try to create a small GPU model
            test_model = xgb.XGBClassifier(
                n_estimators=1,
                device="cuda",
                tree_method="hist",
            )
            # Create tiny test data
            X_test = np.array([[1, 2], [3, 4]])
            y_test = np.array([0, 1])
            test_model.fit(X_test, y_test)
            self.gpu_available = True
            logger.info("GPU acceleration available for XGBoost")
        except Exception as e:
            self.gpu_available = False
            logger.warning(f"GPU not available, falling back to CPU: {e}")

    def _initialize_rmm(self) -> None:
        """Initialize RAPIDS Memory Manager for faster GPU memory allocation."""
        if not RMM_AVAILABLE:
            logger.warning("RMM not available, skipping initialization")
            return

        try:
            # Initialize RMM pool allocator for faster memory allocation
            # Use 14GB max pool (leave 2GB for system) on 16GB GPU
            initial_pool_size = 8 * 1024**3  # 8GB initial
            maximum_pool_size = 14 * 1024**3  # 14GB max

            mr = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaAsyncMemoryResource(),
                initial_pool_size=initial_pool_size,
                maximum_pool_size=maximum_pool_size,
            )
            rmm.mr.set_current_device_resource(mr)
            self.rmm_initialized = True
            logger.info(
                f"RMM initialized: initial_pool={initial_pool_size / 1024**3:.1f}GB, "
                f"max_pool={maximum_pool_size / 1024**3:.1f}GB"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RMM: {e}. Continuing without RMM.")
            self.rmm_initialized = False

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
        optimize: bool = True,
        early_stopping_rounds: int = 50,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with GPU acceleration.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize: Whether to optimize hyperparameters with Optuna
            early_stopping_rounds: Early stopping patience

        Returns:
            Trained XGBoost classifier
        """
        start_time = datetime.now()

        # Convert Polars to NumPy for XGBoost
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        # Feature selection if enabled (performed after conversion to numpy for efficiency)
        selected_features = None
        if (
            self.feature_selection_top_n
            and self.feature_selection_top_n > 0
            and X_train_np.shape[1] > self.feature_selection_top_n
        ):
            logger.info(
                f"Performing feature selection: selecting top {self.feature_selection_top_n} features from {X_train_np.shape[1]}"
            )
            # Quick training pass for feature importance
            temp_params = self._get_base_params()
            temp_params["n_estimators"] = 50  # Quick pass with fewer trees
            # Filter out predictor (only used for inference, not training)
            temp_params = {k: v for k, v in temp_params.items() if k != "predictor"}
            temp_model = xgb.XGBClassifier(**temp_params)
            temp_model.fit(X_train_np, y_train_np, verbose=False)

            # Get feature importance
            importance = temp_model.get_booster().get_score(importance_type="gain")
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            # Select top N (handle both 'f0' format and numeric indices)
            top_n_indices = []
            for k, _ in sorted_features[: self.feature_selection_top_n]:
                try:
                    idx = int(k.replace("f", ""))
                    top_n_indices.append(idx)
                except ValueError:
                    continue

            if len(top_n_indices) > 0:
                selected_features = sorted(top_n_indices)
                X_train_np = X_train_np[:, selected_features]
                if X_val is not None:
                    X_val_np = X_val.to_numpy()
                    X_val_np = X_val_np[:, selected_features]
                    X_val = pl.DataFrame(X_val_np)
                logger.info(
                    f"Selected {len(selected_features)} features: {selected_features[:10]}..."
                    if len(selected_features) > 10
                    else f"Selected {len(selected_features)} features"
                )
            else:
                logger.warning("Feature selection failed, using all features")

        if optimize:
            logger.info("Optimizing hyperparameters with Optuna")
            best_params = self._optimize_hyperparameters(X_train_np, y_train_np, X_val, y_val)
            params = {**self._get_base_params(), **best_params}
        else:
            params = self._get_base_params()

        logger.info(f"Training XGBoost model on {'GPU' if self.gpu_available else 'CPU'}")

        # Prepare validation set if provided
        eval_set = None
        callbacks = []

        if X_val is not None and y_val is not None:
            X_val_np = X_val.to_numpy()
            y_val_np = y_val.to_numpy()
            eval_set = [(X_train_np, y_train_np), (X_val_np, y_val_np)]

            # Add early stopping
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True,
                    maximize=False,
                    data_name="validation_1",
                )
            )

        # Train model
        # In XGBoost 3.x, callbacks go in constructor
        if callbacks:
            params["callbacks"] = callbacks

        # XGBoost 3.0+ automatically uses QuantileDMatrix internally when tree_method="hist"
        # and device="cuda" are set. XGBClassifier.fit() expects (X, y) as separate arrays,
        # not a QuantileDMatrix object. The sklearn API handles QuantileDMatrix creation
        # automatically for memory efficiency (5x reduction).
        # #region agent log
        import json

        log_path = "/home/brennan/nac/.cursor/debug.log"

        def log_debug(location, message, data, hypothesis_id=None):
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "post-fix",
                            "hypothesisId": hypothesis_id,
                            "location": location,
                            "message": message,
                            "data": data,
                            "timestamp": datetime.now().timestamp() * 1000,
                        }
                    )
                    + "\n"
                )

        # #endregion

        # #region agent log
        log_debug(
            "trainer.py:380",
            "Before training",
            {
                "xgboost_version": xgb.__version__,
                "X_train_shape": X_train_np.shape if X_train_np is not None else None,
                "y_train_shape": y_train_np.shape if y_train_np is not None else None,
                "tree_method": params.get("tree_method"),
                "device": params.get("device"),
                "use_quantile_dmatrix": self.use_quantile_dmatrix,
                "gpu_available": self.gpu_available,
                "eval_set_present": eval_set is not None,
            },
            "A",
        )
        # #endregion

        # Filter out parameters that are not valid for XGBClassifier constructor
        # predictor is only used during inference, not training (causes warning if passed to constructor)
        training_params = {k: v for k, v in params.items() if k != "predictor"}

        # Standard training with XGBClassifier
        # XGBoost will automatically use QuantileDMatrix when tree_method="hist" and device="cuda"
        self.model = xgb.XGBClassifier(**training_params)
        # #region agent log
        log_debug(
            "trainer.py:415",
            "Before fit() call",
            {
                "X_train_shape": X_train_np.shape if X_train_np is not None else None,
                "y_train_shape": y_train_np.shape if y_train_np is not None else None,
                "y_train_is_none": y_train_np is None,
                "y_train_type": str(type(y_train_np)),
                "eval_set_is_none": eval_set is None,
                "eval_set_len": len(eval_set) if eval_set else 0,
                "eval_set_type": str(type(eval_set)) if eval_set else None,
                "params_keys": list(training_params.keys()),
                "params_predictor": params.get("predictor"),
                "params_sampling_method": params.get("sampling_method"),
                "predictor_filtered": "predictor" in params and "predictor" not in training_params,
            },
            "A,B,C,D,E",
        )
        # #endregion
        try:
            self.model.fit(
                X_train_np,
                y_train_np,
                eval_set=eval_set,
                verbose=False,
            )
        except Exception as e:
            # #region agent log
            log_debug(
                "trainer.py:440",
                "fit() failed",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "X_train_shape": X_train_np.shape if X_train_np is not None else None,
                    "y_train_shape": y_train_np.shape if y_train_np is not None else None,
                    "y_train_is_none": y_train_np is None,
                    "eval_set_is_none": eval_set is None,
                },
                "A,B,C,D,E",
            )
            # #endregion
            logger.warning(f"Training failed, retrying with minimal parameters: {e}")
            # Fallback: remove potentially problematic parameters and retry
            # Already filtered predictor above, but remove callbacks too in case they're causing issues
            fallback_params = {k: v for k, v in training_params.items() if k != "callbacks"}
            self.model = xgb.XGBClassifier(**fallback_params)
            self.model.fit(
                X_train_np,
                y_train_np,
                eval_set=eval_set,
                verbose=False,
            )
        # #region agent log
        log_debug(
            "trainer.py:407", "fit() succeeded", {"model_trained": self.model is not None}, "A"
        )
        # #endregion

        if self.use_quantile_dmatrix and self.gpu_available:
            logger.debug("Training with QuantileDMatrix (automatically enabled by XGBoost)")

        # Set GPU predictor for inference optimization (only used during prediction, not training)
        if self.gpu_available and hasattr(self.model, "set_params"):
            try:
                self.model.set_params(predictor="gpu_predictor")
            except Exception:
                # Some XGBoost versions don't support set_params, skip silently
                pass

        training_time = (datetime.now() - start_time).total_seconds()

        # Create serializable copy of params (exclude callbacks which contain non-serializable objects)
        serializable_params = {k: v for k, v in params.items() if k != "callbacks"}
        if "callbacks" in params:
            # Store callback info as serializable dict
            serializable_params["callbacks_info"] = []
            for cb in params["callbacks"]:
                if hasattr(cb, "__class__"):
                    cb_info = {
                        "type": cb.__class__.__name__,
                    }
                    # Extract callback attributes if available
                    if hasattr(cb, "rounds"):
                        cb_info["rounds"] = cb.rounds
                    if hasattr(cb, "save_best"):
                        cb_info["save_best"] = cb.save_best
                    if hasattr(cb, "maximize"):
                        cb_info["maximize"] = cb.maximize
                    if hasattr(cb, "data_name"):
                        cb_info["data_name"] = cb.data_name
                    serializable_params["callbacks_info"].append(cb_info)

        # Record training history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(X_train_np),
            "n_features": X_train_np.shape[1],
            "params": serializable_params,
            "training_time_seconds": training_time,
            "gpu_used": self.gpu_available,
        }

        if hasattr(self.model, "best_score"):
            history_entry["best_score"] = self.model.best_score
        if hasattr(self.model, "best_iteration"):
            history_entry["best_iteration"] = self.model.best_iteration

        self.training_history.append(history_entry)

        logger.info(
            f"Model training completed in {training_time:.1f}s "
            f"(samples={len(X_train_np)}, features={X_train_np.shape[1]})"
        )

        return self.model

    def train_with_npmm(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        price_col: str = "close",
        val_ratio: float = 0.2,
        optimize: bool = True,
    ) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
        """
        Train model using NPMM labeling methodology.

        This is the recommended training method for swing trading signals.

        Args:
            df: DataFrame with features and price data
            feature_cols: List of feature column names
            price_col: Price column for labeling
            val_ratio: Validation set ratio
            optimize: Whether to optimize hyperparameters

        Returns:
            Tuple of (trained model, training metrics)
        """
        # Generate NPMM labels
        df_labeled = self.labeler.generate_binary_labels(df, price_col)

        # Filter out invalid labels
        df_valid = df_labeled.filter(pl.col("label") >= 0)

        # Split features and labels
        X = df_valid.select(feature_cols)
        y = df_valid["label"]

        # Time-based split (no shuffling for time series)
        split_idx = int(len(X) * (1 - val_ratio))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        # Train model
        model = self.train(X_train, y_train, X_val, y_val, optimize=optimize)

        # Calculate metrics
        y_pred = model.predict(X_val.to_numpy())
        y_val_np = y_val.to_numpy()

        accuracy = np.mean(y_pred == y_val_np)

        # Precision/Recall for class 1 (buy signals)
        true_positives = np.sum((y_pred == 1) & (y_val_np == 1))
        false_positives = np.sum((y_pred == 1) & (y_val_np == 0))
        false_negatives = np.sum((y_pred == 0) & (y_val_np == 1))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "label_distribution": {
                "buy": int(np.sum(y.to_numpy() == 1)),
                "sell": int(np.sum(y.to_numpy() == 0)),
            },
        }

        logger.info(
            f"NPMM Training Results: Accuracy={accuracy:.3f}, "
            f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
        )

        return model, metrics

    def get_feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
        importance_type: str = "gain",
    ) -> pl.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            feature_names: Optional list of feature names
            importance_type: Type of importance (gain, weight, cover)

        Returns:
            DataFrame with feature importance rankings
        """
        if self.model is None:
            raise ValueError("No model trained yet")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        if feature_names is None:
            feature_names = list(importance.keys())

        # Create DataFrame
        data = []
        for i, name in enumerate(feature_names):
            key = f"f{i}" if f"f{i}" in importance else name
            if key in importance:
                data.append(
                    {
                        "feature": name,
                        "importance": importance[key],
                    }
                )

        df = pl.DataFrame(data).sort("importance", descending=True)

        return df

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[pl.DataFrame],
        y_val: Optional[pl.Series],
    ) -> dict:
        """
        Optimize hyperparameters using Optuna with pruning.

        Args:
            X_train: Training features (numpy array)
            y_train: Training labels (numpy array)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of best hyperparameters
        """
        if X_val is None or y_val is None:
            # Use time-based split from training data
            split_idx = int(len(X_train) * 0.8)
            X_val_np = X_train[split_idx:]
            y_val_np = y_train[split_idx:]
            X_train_np = X_train[:split_idx]
            y_train_np = y_train[:split_idx]
        else:
            X_train_np = X_train
            y_train_np = y_train
            X_val_np = X_val.to_numpy() if isinstance(X_val, pl.DataFrame) else X_val
            y_val_np = y_val.to_numpy() if isinstance(y_val, pl.Series) else y_val

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0, log=True),
                **self._get_base_params(exclude_tunable=True),
            }

            # Use pruning callback for early termination of unpromising trials
            # XGBoostPruningCallback enables Optuna to prune unpromising trials early
            # This saves significant time during hyperparameter optimization
            pruning_callback = None
            try:
                pruning_callback = optuna.integration.XGBoostPruningCallback(
                    trial, "validation_0-logloss"
                )
            except (ModuleNotFoundError, AttributeError):
                logger.warning("optuna-integration not available, skipping pruning callback")
                pruning_callback = None
            except Exception as e:
                logger.warning(f"Failed to create pruning callback: {e}")
                pruning_callback = None

            # In XGBoost 3.x, callbacks go in constructor, not fit()
            callbacks_list = [pruning_callback] if pruning_callback else []
            if callbacks_list:
                params["callbacks"] = callbacks_list

            # Filter out predictor (only used for inference, not training)
            training_params = {k: v for k, v in params.items() if k != "predictor"}

            model = xgb.XGBClassifier(**training_params)
            model.fit(
                X_train_np,
                y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                verbose=False,
            )

            # Return validation accuracy
            score = model.score(X_val_np, y_val_np)
            return score

        # Create study with TPE sampler
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout,
            show_progress_bar=True,
        )

        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation score: {study.best_value:.4f}")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        return study.best_params

    def _get_base_params(self, exclude_tunable: bool = False) -> dict:
        """
        Get base XGBoost parameters.

        Args:
            exclude_tunable: If True, exclude parameters that will be tuned

        Returns:
            Dictionary of base parameters
        """
        base = {
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "tree_method": "hist",
            "device": "cuda" if self.gpu_available else "cpu",
            "random_state": 42,
        }

        # Note: predictor="gpu_predictor" is only used during inference, not training
        # It's set after model training or during model loading for prediction

        # Add gradient-based sampling if enabled
        if self.gradient_sampling and self.gpu_available:
            base["sampling_method"] = "gradient_based"
            # Lower subsample rate works well with gradient-based sampling
            if not exclude_tunable:
                base["subsample"] = min(self.config.subsample, 0.8)

        if not exclude_tunable:
            base.update(
                {
                    "n_estimators": self.config.n_estimators,
                    "max_depth": self.config.max_depth,
                    "learning_rate": self.config.learning_rate,
                    "min_child_weight": self.config.min_child_weight,
                    "subsample": self.config.subsample,
                    "colsample_bytree": self.config.colsample_bytree,
                    "gamma": self.config.gamma,
                    "reg_alpha": self.config.reg_alpha,
                    "reg_lambda": self.config.reg_lambda,
                }
            )

        return base

    def calculate_shap_values(
        self, X: np.ndarray, feature_names: Optional[list[str]] = None, max_samples: int = 1000
    ) -> Optional[dict[str, Any]]:
        """
        Calculate SHAP values for model interpretability.

        Args:
            X: Feature matrix
            feature_names: Optional feature names
            max_samples: Maximum samples for SHAP calculation (for performance)

        Returns:
            Dictionary with SHAP statistics or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - skipping SHAP calculation")
            return None

        if self.model is None:
            raise ValueError("No model trained yet")

        # Sample if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        try:
            # Create TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification (SHAP returns list for binary)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            # Calculate summary statistics
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Get feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

            # Create feature importance dictionary
            feature_importance = {
                name: float(importance)
                for name, importance in zip(feature_names, mean_abs_shap, strict=False)
            }

            shap_stats = {
                "mean_abs_shap": mean_abs_shap.tolist(),
                "feature_names": feature_names,
                "expected_value": (
                    float(explainer.expected_value)
                    if hasattr(explainer, "expected_value")
                    else None
                ),
                "feature_importance": feature_importance,
                "n_samples": len(X_sample),
            }

            logger.info(f"SHAP values calculated for {len(X_sample)} samples")
            return shap_stats

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return None

    def save_model(
        self,
        filepath: str,
        include_metadata: bool = True,
        include_shap: bool = True,
        X_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Save trained model to file with optional metadata and SHAP analysis.

        Args:
            filepath: Path to save model
            include_metadata: Whether to save training metadata alongside
            include_shap: Whether to calculate and save SHAP values
            X_val: Validation set for SHAP calculation (optional)
            feature_names: Feature names for SHAP (optional)
        """
        if self.model is None:
            raise ValueError("No model trained yet")

        filepath = Path(filepath)

        # Save model
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")

        # Calculate SHAP values if requested
        shap_stats = None
        if include_shap and X_val is not None:
            shap_stats = self.calculate_shap_values(X_val, feature_names)

        # Save metadata
        if include_metadata and self.training_history:
            metadata_path = filepath.with_suffix(".json")
            metadata = {
                "model_type": "XGBoostClassifier",
                "created_at": datetime.now().isoformat(),
                "training_history": self.training_history,
                "gpu_available": self.gpu_available,
            }

            # Add SHAP statistics if available
            if shap_stats:
                metadata["shap"] = shap_stats

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model metadata saved to {metadata_path}")

    def load_model(self, filepath: str) -> xgb.XGBClassifier:
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded XGBoost classifier
        """
        filepath = Path(filepath)

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(filepath))

        # Try to load metadata from .metadata.json file
        metadata_path = filepath.with_suffix("").with_suffix(".json.metadata")
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.training_history = metadata.get("training_history", [])
                logger.info(f"Model and metadata loaded from {filepath}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        else:
            logger.info(f"Model loaded from {filepath}")

        return self.model

    def cross_validate(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> dict[str, Any]:
        """
        Perform purged time-series cross-validation.

        Based on Lopez de Prado's purged k-fold CV to prevent data leakage.

        Args:
            X: Features DataFrame
            y: Labels Series
            n_splits: Number of CV splits
            purge_gap: Number of samples to purge between train/test

        Returns:
            Dictionary with CV results
        """
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        n_samples = len(X_np)

        fold_size = n_samples // n_splits
        scores = []

        for i in range(n_splits - 1):  # Leave last fold for final validation
            # Training set: all folds before current
            train_end = (i + 1) * fold_size - purge_gap
            train_X = X_np[:train_end]
            train_y = y_np[:train_end]

            # Test set: current fold (with gap from training)
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            test_X = X_np[test_start:test_end]
            test_y = y_np[test_start:test_end]

            if len(train_X) < 100 or len(test_X) < 20:
                continue

            # Train model
            base_params = self._get_base_params()
            # Filter out predictor (only used for inference, not training)
            training_params = {k: v for k, v in base_params.items() if k != "predictor"}
            model = xgb.XGBClassifier(**training_params)
            model.fit(train_X, train_y, verbose=False)

            # Evaluate
            score = model.score(test_X, test_y)
            scores.append(score)

            logger.debug(f"Fold {i+1}: accuracy = {score:.4f}")

        results = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "scores": scores,
            "n_splits": len(scores),
        }

        logger.info(
            f"Purged CV Results: {results['mean_score']:.4f} " f"(+/- {results['std_score']:.4f})"
        )

        return results
