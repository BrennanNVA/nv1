"""Model inference and prediction aggregation."""

import logging
from typing import Optional

import polars as pl
import xgboost as xgb

logger = logging.getLogger(__name__)


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
