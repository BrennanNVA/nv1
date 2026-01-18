"""Model inference and prediction aggregation."""

import logging
from pathlib import Path
from typing import Optional

import polars as pl
import xgboost as xgb

logger = logging.getLogger(__name__)

# Import master ensemble model
try:
    from .master_ensemble import MasterEnsembleModel
except ImportError:
    MasterEnsembleModel = None
    logger.warning("MasterEnsembleModel not available")


class ModelPredictor:
    """Make predictions using trained XGBoost models."""

    def __init__(self, model: Optional[xgb.XGBClassifier] = None) -> None:
        """
        Initialize model predictor.

        Args:
            model: Trained XGBoost model (can be loaded later)
        """
        self.model = model
        logger.info("ModelPredictor initialized")

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        Make binary predictions (0 or 1).

        Args:
            X: Feature DataFrame

        Returns:
            Series with predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("No model loaded")

        X_np = X.to_numpy()
        predictions = self.model.predict(X_np)

        return pl.Series(predictions)

    def predict_proba(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Predict class probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with probability columns ('prob_0', 'prob_1')
        """
        if self.model is None:
            raise ValueError("No model loaded")

        X_np = X.to_numpy()
        probabilities = self.model.predict_proba(X_np)

        return pl.DataFrame(
            {
                "prob_0": probabilities[:, 0],
                "prob_1": probabilities[:, 1],
            }
        )

    def predict_with_confidence(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Make predictions with confidence scores.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with 'prediction' and 'confidence' columns
        """
        proba_df = self.predict_proba(X)
        predictions = self.predict(X)

        # Confidence is the maximum probability
        confidence = proba_df.select(pl.max_horizontal("prob_0", "prob_1")).rename(
            {"max": "confidence"}
        )

        return pl.DataFrame(
            {
                "prediction": predictions,
                "confidence": confidence["confidence"],
            }
        )

    def load_model(self, filepath: str) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to model file
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class ModelRegistry:
    """Registry for managing symbol-specific model loading and caching.

    Institutional-grade approach: Individual models per symbol for better accuracy.
    Models are loaded on-demand and cached to avoid repeated file I/O.

    This follows the practice of major quant funds (Renaissance, Two Sigma, etc.)
    which train separate models per asset to capture symbol-specific patterns.
    """

    def __init__(self, model_dir: Path) -> None:
        """
        Initialize model registry.

        Args:
            model_dir: Directory containing model files (e.g., project_root / "models")
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self._cache: dict[str, ModelPredictor] = {}  # symbol -> predictor
        logger.info(f"ModelRegistry initialized (model_dir: {self.model_dir})")

    def get_predictor(self, symbol: str) -> Optional[ModelPredictor]:
        """
        Get predictor for a symbol, loading from cache or file if needed.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            ModelPredictor instance if model found, None otherwise
        """
        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        # Find latest model file for this symbol
        # Pattern: {SYMBOL}_{DATE}.json (e.g., AAPL_20250117.json)
        model_files = list(self.model_dir.glob(f"{symbol}_*.json"))
        if not model_files:
            logger.debug(f"No model found for {symbol}")
            return None

        # Get latest model by modification time
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        try:
            predictor = ModelPredictor()
            predictor.load_model(str(latest_model))
            self._cache[symbol] = predictor
            logger.debug(f"Loaded model for {symbol}: {latest_model.name}")
            return predictor
        except Exception as e:
            logger.warning(f"Failed to load model for {symbol} ({latest_model}): {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the model cache (useful for memory management)."""
        self._cache.clear()
        logger.debug("Model cache cleared")

    def get_available_symbols(self) -> list[str]:
        """
        Get list of symbols that have trained models available.

        Returns:
            List of symbol strings
        """
        model_files = list(self.model_dir.glob("*_*.json"))
        symbols = set()
        for model_file in model_files:
            # Extract symbol from filename (e.g., "AAPL_20250117.json" -> "AAPL")
            # Skip master_model files
            if "master_model" in model_file.stem:
                continue
            parts = model_file.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                symbols.add(symbol)
        return sorted(list(symbols))

    def get_master_model(self) -> Optional["MasterEnsembleModel"]:
        """
        Get master ensemble model if available.

        Returns:
            MasterEnsembleModel instance if found, None otherwise
        """
        if MasterEnsembleModel is None:
            return None

        # Find latest master model file
        master_files = list(self.model_dir.glob("master_model_*.json"))
        if not master_files:
            logger.debug("No master model found")
            return None

        # Get latest master model by modification time
        latest_master = max(master_files, key=lambda p: p.stat().st_mtime)

        try:
            master_model = MasterEnsembleModel(self.model_dir)
            master_model.load(latest_master)
            logger.debug(f"Loaded master model: {latest_master.name}")
            return master_model
        except Exception as e:
            logger.warning(f"Failed to load master model ({latest_master}): {e}")
            return None
