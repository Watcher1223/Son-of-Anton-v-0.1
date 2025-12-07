"""
Logistic Regression Model for Schema Extraction Prediction

Baseline classifier using multinomial logistic regression.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier:
    """
    Logistic Regression classifier wrapper for schema extraction prediction.
    
    Provides consistent interface for training, prediction, and model persistence.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        class_weight: Optional[str] = 'balanced',
        random_state: int = 42,
        config: Optional[Dict] = None
    ):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
            solver: Optimization algorithm
            class_weight: Class weight strategy ('balanced' or None)
            random_state: Random seed for reproducibility
            config: Additional configuration dictionary
        """
        self.config = config or {}
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
            multi_class='multinomial',
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names: list = []
        self.class_labels: list = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        class_labels: Optional[list] = None
    ) -> 'LogisticRegressionClassifier':
        """
        Train the logistic regression model.
        
        Args:
            X: Training feature matrix
            y: Training labels
            feature_names: List of feature names
            class_labels: List of class labels
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Logistic Regression on {X.shape[0]} samples, {X.shape[1]} features")
        
        self.feature_names = feature_names or []
        self.class_labels = class_labels or []
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Training complete")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probability matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_macro'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        results = {
            'cv_scores': scores.tolist(),
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
        }
        
        logger.info(f"CV {scoring}: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'f1_macro'
    ) -> Tuple['LogisticRegressionClassifier', Dict]:
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Labels
            param_grid: Parameter grid (uses default if None)
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Tuple of (best model, search results)
        """
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga'],
                'class_weight': ['balanced', None],
            }
        
        logger.info(f"Running hyperparameter search with {cv}-fold CV...")
        
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'params': grid_search.cv_results_['params'],
            }
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return self, results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on coefficient magnitudes.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Average absolute coefficient across classes
        coef_importance = np.abs(self.model.coef_).mean(axis=0)
        
        if self.feature_names:
            importance = dict(zip(self.feature_names, coef_importance))
        else:
            importance = {f'feature_{i}': coef_importance[i] for i in range(len(coef_importance))}
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def save(self, path: Path) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'class_labels': self.class_labels,
            'is_fitted': self.is_fitted,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'LogisticRegressionClassifier':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded classifier instance
        """
        path = Path(path)
        model_data = joblib.load(path)
        
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.class_labels = model_data['class_labels']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        
        return instance


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 3, 100)
    
    # Train model
    model = LogisticRegressionClassifier()
    model.fit(X, y, feature_names=[f'feat_{i}' for i in range(20)])
    
    # Predict
    y_pred = model.predict(X)
    print(f"Predictions shape: {y_pred.shape}")
    
    # Cross-validate
    cv_results = model.cross_validate(X, y)
    
    # Feature importance
    importance = model.get_feature_importance()
    print("\nTop 5 features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:5]):
        print(f"  {feat}: {imp:.4f}")

