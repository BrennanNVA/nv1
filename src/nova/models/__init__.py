"""ML models: training and prediction."""

from .predictor import ModelPredictor, ModelRegistry
from .trainer import ModelTrainer

try:
    from .master_ensemble import MasterEnsembleModel

    __all__ = ["ModelTrainer", "ModelPredictor", "ModelRegistry", "MasterEnsembleModel"]
except ImportError:
    __all__ = ["ModelTrainer", "ModelPredictor", "ModelRegistry"]
