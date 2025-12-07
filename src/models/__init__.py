"""
Machine Learning Models for API Schema Extraction Prediction

This module contains:
- preprocessor: Data loading, encoding, scaling, and train/test splitting
- logistic_regression: Logistic Regression classifier
- xgboost_model: XGBoost classifier
- neural_network: PyTorch feedforward neural network
- evaluation: Metrics and evaluation utilities
- explainability: SHAP-based feature importance
"""

__version__ = "0.1.0"

from .preprocessor import DataPreprocessor
from .evaluation import ModelEvaluator

__all__ = [
    "DataPreprocessor",
    "ModelEvaluator",
]

