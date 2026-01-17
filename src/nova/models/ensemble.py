"""Ensemble methods: Stacking and Blending for model combination.

Research-backed implementation based on:
- Stacking with meta-learner (linear or XGBoost)
- Out-of-fold predictions to prevent leakage
- Multiple base models (XGBoost, LightGBM, CatBoost)
- Blending as simpler alternative

Research Finding: Ensembles show 3-5% higher returns in 2024-2025 studies.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class MetaLearnerType(Enum):
    """Type of meta-learner for stacking."""

    LINEAR = "linear"
    RIDGE = "ridge"
    XGBOOST = "xgboost"


@dataclass
class BaseModel:
    """Base model wrapper for ensemble."""

    name: str
    model: Any
    weight: float = 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        elif hasattr(self.model, "predict_proba"):
            # For classification, use probability
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]  # Return positive class probability
            return proba
        else:
            raise ValueError(f"Model {self.name} does not have predict method")


class EnsembleStacker:
    """Stacking ensemble with meta-learner.

    Implements:
    - Out-of-fold predictions for base models
    - Meta-learner training on OOF predictions
    - Support for multiple base models
    - Cross-validation for stacking
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        meta_learner_type: MetaLearnerType = MetaLearnerType.RIDGE,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Initialize ensemble stacker.

        Args:
            base_models: List of base models
            meta_learner_type: Type of meta-learner
            n_folds: Number of folds for cross-validation
            random_state: Random seed
        """
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.n_folds = n_folds
        self.random_state = random_state

        self.meta_learner: Optional[Any] = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.is_fitted = False

        logger.info(
            f"EnsembleStacker initialized: {len(base_models)} base models, "
            f"meta_learner={meta_learner_type.value}, n_folds={n_folds}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit ensemble using stacking with out-of-fold predictions.

        Args:
            X: Feature matrix
            y: Target vector
            sample_weight: Optional sample weights
        """
        n_samples = len(X)
        n_models = len(self.base_models)

        # Initialize OOF predictions matrix
        oof_predictions = np.zeros((n_samples, n_models))

        # K-fold cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        logger.info("Training base models with out-of-fold predictions...")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.debug(f"Fold {fold_idx + 1}/{self.n_folds}")

            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # Train each base model on fold training data
            for model_idx, base_model in enumerate(self.base_models):
                logger.debug(f"  Training {base_model.name}...")

                # Train base model
                if hasattr(base_model.model, "fit"):
                    if sample_weight is not None:
                        base_model.model.fit(
                            X_train_fold, y_train_fold, sample_weight=sample_weight[train_idx]
                        )
                    else:
                        base_model.model.fit(X_train_fold, y_train_fold)

                # Predict on validation fold (out-of-fold)
                val_pred = base_model.predict(X_val_fold)
                oof_predictions[val_idx, model_idx] = val_pred

        self.oof_predictions = oof_predictions

        # Train meta-learner on OOF predictions
        logger.info("Training meta-learner...")
        self.meta_learner = self._create_meta_learner()

        if sample_weight is not None:
            self.meta_learner.fit(oof_predictions, y, sample_weight=sample_weight)
        else:
            self.meta_learner.fit(oof_predictions, y)

        # Retrain base models on full dataset
        logger.info("Retraining base models on full dataset...")
        for base_model in self.base_models:
            if hasattr(base_model.model, "fit"):
                if sample_weight is not None:
                    base_model.model.fit(X, y, sample_weight=sample_weight)
                else:
                    base_model.model.fit(X, y)

        self.is_fitted = True
        logger.info("Ensemble stacking complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacked ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predictions from meta-learner
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all base models
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for model_idx, base_model in enumerate(self.base_models):
            base_predictions[:, model_idx] = base_model.predict(X)

        # Meta-learner combines base predictions
        meta_predictions = self.meta_learner.predict(base_predictions)

        return meta_predictions

    def _create_meta_learner(self) -> Any:
        """Create meta-learner based on type."""
        if self.meta_learner_type == MetaLearnerType.LINEAR:
            return LinearRegression()
        elif self.meta_learner_type == MetaLearnerType.RIDGE:
            return Ridge(alpha=1.0)
        elif self.meta_learner_type == MetaLearnerType.XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance from meta-learner (if available).

        Returns:
            Dictionary mapping base model names to importance scores
        """
        if not self.is_fitted or self.meta_learner is None:
            return {}

        importances = {}

        if hasattr(self.meta_learner, "coef_"):
            # Linear/Ridge regression coefficients
            coefs = self.meta_learner.coef_
            if coefs.ndim == 1:
                for i, base_model in enumerate(self.base_models):
                    importances[base_model.name] = float(abs(coefs[i]))
        elif hasattr(self.meta_learner, "feature_importances_"):
            # XGBoost feature importance
            importances_array = self.meta_learner.feature_importances_
            for i, base_model in enumerate(self.base_models):
                importances[base_model.name] = float(importances_array[i])

        return importances


class EnsembleBlender:
    """Simple blending ensemble (weighted average).

    Simpler alternative to stacking - no meta-learner needed.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        weights: Optional[list[float]] = None,
    ) -> None:
        """
        Initialize ensemble blender.

        Args:
            base_models: List of base models
            weights: Optional weights for each model (default: equal weights)
        """
        self.base_models = base_models

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(base_models)] * len(base_models)
        else:
            if len(weights) != len(base_models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        # Update base model weights
        for i, base_model in enumerate(self.base_models):
            base_model.weight = self.weights[i]

        logger.info(
            f"EnsembleBlender initialized: {len(base_models)} models, " f"weights={self.weights}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit all base models.

        Args:
            X: Feature matrix
            y: Target vector
            sample_weight: Optional sample weights
        """
        logger.info("Training base models for blending...")

        for base_model in self.base_models:
            logger.debug(f"Training {base_model.name}...")
            if hasattr(base_model.model, "fit"):
                if sample_weight is not None:
                    base_model.model.fit(X, y, sample_weight=sample_weight)
                else:
                    base_model.model.fit(X, y)

        logger.info("Blending ensemble training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted average of base models.

        Args:
            X: Feature matrix

        Returns:
            Blended predictions
        """
        predictions = []

        for base_model in self.base_models:
            pred = base_model.predict(X)
            predictions.append(pred)

        # Weighted average
        predictions_array = np.array(predictions)  # Shape: (n_models, n_samples)
        weights_array = np.array(self.weights).reshape(-1, 1)  # Shape: (n_models, 1)

        blended = np.sum(predictions_array * weights_array, axis=0)

        return blended

    def set_weights(self, weights: list[float]) -> None:
        """
        Update model weights.

        Args:
            weights: New weights for each model
        """
        if len(weights) != len(self.base_models):
            raise ValueError("Number of weights must match number of models")

        # Normalize
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # Update base model weights
        for i, base_model in enumerate(self.base_models):
            base_model.weight = self.weights[i]

        logger.info(f"Updated weights: {self.weights}")


def create_xgboost_model(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> BaseModel:
    """Create XGBoost base model."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        **kwargs,
    )
    return BaseModel(name="XGBoost", model=model)


def create_lightgbm_model(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> BaseModel:
    """Create LightGBM base model (if available)."""
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs,
        )
        return BaseModel(name="LightGBM", model=model)
    except ImportError:
        logger.warning("LightGBM not available - install with: pip install lightgbm")
        return None


def create_catboost_model(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> BaseModel:
    """Create CatBoost base model (if available)."""
    try:
        import catboost as cb

        model = cb.CatBoostRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )
        return BaseModel(name="CatBoost", model=model)
    except ImportError:
        logger.warning("CatBoost not available - install with: pip install catboost")
        return None
