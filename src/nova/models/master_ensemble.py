"""Master Ensemble Model - Renaissance-style unified model for combining individual symbol models.

This implements the "major model" that learns how to optimally combine predictions from
all individual symbol models, capturing cross-symbol patterns and adapting weights dynamically.

Research-backed approach based on:
- Renaissance Technologies unified ensemble system
- Meta-learning for model combination
- Cross-symbol pattern recognition
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import xgboost as xgb

logger = logging.getLogger(__name__)


class MasterEnsembleModel:
    """Master ensemble model that learns to combine individual symbol model predictions.

    This is the "major model" that gets refined over time, learning:
    - Optimal weights for each symbol model
    - Cross-symbol patterns (e.g., tech sector correlations)
    - Market regime-specific model performance
    - Dynamic adaptation to changing market conditions
    """

    def __init__(
        self,
        model_dir: Path,
        meta_learner_type: str = "xgboost",
        use_cross_symbol_features: bool = True,
    ) -> None:
        """
        Initialize master ensemble model.

        Args:
            model_dir: Directory containing individual symbol models and where master model will be saved
            meta_learner_type: Type of meta-learner ("xgboost", "ridge", "linear")
            use_cross_symbol_features: Whether to include cross-symbol features (sector correlations, etc.)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.meta_learner_type = meta_learner_type
        self.use_cross_symbol_features = use_cross_symbol_features

        self.meta_model: Optional[xgb.XGBRegressor] = None
        self.symbol_list: list[str] = []
        self.is_fitted = False

        logger.info(
            f"MasterEnsembleModel initialized: meta_learner={meta_learner_type}, "
            f"cross_symbol_features={use_cross_symbol_features}"
        )

    def collect_predictions(
        self,
        symbol: str,
        predictor: Any,
        features_df: pl.DataFrame,
        feature_names: list[str],
    ) -> Optional[dict[str, float]]:
        """
        Collect prediction from an individual symbol model.

        Args:
            symbol: Stock symbol
            predictor: ModelPredictor instance for this symbol
            features_df: Feature DataFrame for prediction
            feature_names: List of feature column names

        Returns:
            Dictionary with prediction, confidence, and metadata, or None if failed
        """
        try:
            feature_cols = [col for col in feature_names if col in features_df.columns]
            if not feature_cols:
                return None

            latest_features = features_df.tail(1).select(feature_cols)
            pred_df = predictor.predict_with_confidence(latest_features)

            prediction = float(pred_df["prediction"][0])
            confidence = float(pred_df["confidence"][0])

            # Convert 0/1 to -1/1 scale
            score = prediction * 2 - 1

            return {
                "symbol": symbol,
                "score": score,  # -1 to +1
                "confidence": confidence,  # 0 to 1
                "prediction": prediction,  # 0 or 1
            }
        except Exception as e:
            logger.warning(f"Failed to collect prediction for {symbol}: {e}")
            return None

    def create_meta_features(
        self,
        individual_predictions: dict[str, dict[str, float]],
        market_data: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Create meta-features for master model from individual predictions.

        Features include:
        - Individual model predictions (one per symbol)
        - Individual model confidences
        - Cross-symbol statistics (mean, std, sector correlations)
        - Market regime indicators

        Args:
            individual_predictions: Dict mapping symbol -> {score, confidence, prediction}
            market_data: Optional market data for regime features

        Returns:
            Feature vector for meta-model
        """
        if not individual_predictions:
            raise ValueError("No individual predictions provided")

        features = []

        # Sort symbols for consistent feature ordering
        symbols = sorted(individual_predictions.keys())
        self.symbol_list = symbols

        # 1. Individual model predictions (one per symbol)
        scores = [individual_predictions[s]["score"] for s in symbols]
        confidences = [individual_predictions[s]["confidence"] for s in symbols]

        features.extend(scores)  # N features (one per symbol)
        features.extend(confidences)  # N features (one per symbol)

        # 2. Cross-symbol statistics
        scores_array = np.array(scores)
        confidences_array = np.array(confidences)

        features.append(float(np.mean(scores_array)))  # Mean prediction
        features.append(float(np.std(scores_array)))  # Std of predictions
        features.append(float(np.sum(scores_array > 0)))  # Count of bullish predictions
        features.append(float(np.sum(scores_array < 0)))  # Count of bearish predictions
        features.append(float(np.mean(confidences_array)))  # Mean confidence
        features.append(float(np.std(confidences_array)))  # Std of confidences

        # 3. Sector-based features (if we can identify sectors)
        # Tech sector: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, INTC, NFLX
        tech_symbols = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "AMD",
            "INTC",
            "NFLX",
        }
        finance_symbols = {"JPM", "GS"}
        healthcare_symbols = {"JNJ", "UNH", "PFE"}
        energy_symbols = {"XOM", "CVX"}

        tech_scores = [
            individual_predictions[s]["score"]
            for s in symbols
            if s in tech_symbols and s in individual_predictions
        ]
        finance_scores = [
            individual_predictions[s]["score"]
            for s in symbols
            if s in finance_symbols and s in individual_predictions
        ]
        healthcare_scores = [
            individual_predictions[s]["score"]
            for s in symbols
            if s in healthcare_symbols and s in individual_predictions
        ]
        energy_scores = [
            individual_predictions[s]["score"]
            for s in symbols
            if s in energy_symbols and s in individual_predictions
        ]

        if tech_scores:
            features.append(float(np.mean(tech_scores)))  # Tech sector mean
            features.append(float(np.std(tech_scores)))  # Tech sector std
        else:
            features.extend([0.0, 0.0])

        if finance_scores:
            features.append(float(np.mean(finance_scores)))
            features.append(float(np.std(finance_scores)))
        else:
            features.extend([0.0, 0.0])

        if healthcare_scores:
            features.append(float(np.mean(healthcare_scores)))
            features.append(float(np.std(healthcare_scores)))
        else:
            features.extend([0.0, 0.0])

        if energy_scores:
            features.append(float(np.mean(energy_scores)))
            features.append(float(np.std(energy_scores)))
        else:
            features.extend([0.0, 0.0])

        # 4. Market regime features (if provided)
        if market_data:
            features.append(market_data.get("volatility", 0.0))
            features.append(market_data.get("trend_strength", 0.0))
            if "returns" in market_data and len(market_data["returns"]) > 0:
                returns = np.array(market_data["returns"])
                features.append(float(np.mean(returns)))
                features.append(float(np.std(returns)))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features)

    def train(
        self,
        training_data: list[dict[str, Any]],
        target_symbol: str,
    ) -> dict[str, Any]:
        """
        Train master model on historical predictions and outcomes.

        Args:
            training_data: List of training examples, each containing:
                - individual_predictions: dict of symbol -> prediction dict
                - market_data: market regime data
                - target: actual outcome (future return or label)
            target_symbol: Symbol to predict for (master model predicts per-symbol)

        Returns:
            Training metrics dictionary
        """
        if not training_data:
            raise ValueError("No training data provided")

        logger.info(f"Training master ensemble model for {target_symbol}...")
        logger.info(f"Training examples: {len(training_data)}")

        # Prepare features and targets
        X_list = []
        y_list = []

        for example in training_data:
            individual_predictions = example["individual_predictions"]
            market_data = example.get("market_data")

            # Create meta-features
            try:
                X = self.create_meta_features(individual_predictions, market_data)
                X_list.append(X)

                # Target: future return or label for this symbol
                target = example.get("target", 0.0)
                y_list.append(target)
            except Exception as e:
                logger.warning(f"Skipping training example due to error: {e}")
                continue

        if not X_list:
            raise ValueError("No valid training examples after processing")

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        # Train meta-model
        if self.meta_learner_type == "xgboost":
            self.meta_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        elif self.meta_learner_type == "ridge":
            from sklearn.linear_model import Ridge

            self.meta_model = Ridge(alpha=1.0)
        elif self.meta_learner_type == "linear":
            from sklearn.linear_model import LinearRegression

            self.meta_model = LinearRegression()
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

        self.meta_model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self.meta_model.predict(X_train)
        mse = float(np.mean((y_train - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_train - y_pred)))
        correlation = float(np.corrcoef(y_train, y_pred)[0, 1])

        metrics = {
            "mse": mse,
            "mae": mae,
            "correlation": correlation,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
        }

        logger.info(
            f"Master model training complete: MSE={mse:.4f}, MAE={mae:.4f}, "
            f"Correlation={correlation:.4f}"
        )

        return metrics

    def predict(
        self,
        individual_predictions: dict[str, dict[str, float]],
        market_data: Optional[dict[str, Any]] = None,
        target_symbol: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Make improved prediction using master ensemble model.

        Args:
            individual_predictions: Dict mapping symbol -> prediction dict
            market_data: Optional market regime data
            target_symbol: Symbol to predict for (if None, uses first symbol in predictions)

        Returns:
            Dictionary with improved_score, original_score, confidence, etc.
        """
        if not self.is_fitted or self.meta_model is None:
            raise ValueError("Master model not trained. Call train() first.")

        if not individual_predictions:
            raise ValueError("No individual predictions provided")

        # Create meta-features
        X = self.create_meta_features(individual_predictions, market_data)

        # Get prediction from master model
        improved_score = float(self.meta_model.predict(X.reshape(1, -1))[0])

        # Get original prediction for target symbol
        if target_symbol is None:
            target_symbol = sorted(individual_predictions.keys())[0]

        original_pred = individual_predictions.get(target_symbol)
        if original_pred:
            original_score = original_pred["score"]
            original_confidence = original_pred["confidence"]
        else:
            original_score = 0.0
            original_confidence = 0.0

        return {
            "improved_score": improved_score,
            "original_score": original_score,
            "confidence": original_confidence,
            "target_symbol": target_symbol,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save master model to file.

        Args:
            filepath: Optional path to save model (default: model_dir/master_model_{date}.json)

        Returns:
            Path to saved model file
        """
        if not self.is_fitted or self.meta_model is None:
            raise ValueError("Master model not trained. Cannot save.")

        if filepath is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = self.model_dir / f"master_model_{date_str}.json"

        filepath = Path(filepath)

        # Save XGBoost model
        if isinstance(self.meta_model, xgb.XGBRegressor):
            self.meta_model.save_model(str(filepath))
        else:
            # For sklearn models, use joblib
            import joblib

            joblib.dump(self.meta_model, str(filepath))

        # Save metadata
        metadata_path = filepath.with_suffix(".metadata.json")
        metadata = {
            "meta_learner_type": self.meta_learner_type,
            "symbol_list": self.symbol_list,
            "n_symbols": len(self.symbol_list),
            "use_cross_symbol_features": self.use_cross_symbol_features,
            "trained_date": datetime.now().isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Master model saved to {filepath}")
        return filepath

    def load(self, filepath: Path) -> None:
        """
        Load master model from file.

        Args:
            filepath: Path to model file
        """
        filepath = Path(filepath)

        # Load model
        if self.meta_learner_type == "xgboost" or filepath.suffix == ".json":
            self.meta_model = xgb.XGBRegressor()
            self.meta_model.load_model(str(filepath))
        else:
            import joblib

            self.meta_model = joblib.load(str(filepath))

        # Load metadata
        metadata_path = filepath.with_suffix(".metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.symbol_list = metadata.get("symbol_list", [])
                self.meta_learner_type = metadata.get("meta_learner_type", "xgboost")
                self.use_cross_symbol_features = metadata.get("use_cross_symbol_features", True)

        self.is_fitted = True
        logger.info(f"Master model loaded from {filepath}")
