"""
XGBoost Model for Schema Extraction Prediction

Tree-based classifier that captures feature interactions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost classifier wrapper for schema extraction prediction.
    
    Provides consistent interface for training, prediction, and model persistence.
    XGBoost excels at capturing complex feature interactions.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        config: Optional[Dict] = None
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            config: Additional configuration
        """
        self.config = config or {}
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=0
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.class_labels: List[str] = []
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = None
    ) -> 'XGBoostClassifier':
        """
        Train the XGBoost model.
        
        Args:
            X: Training feature matrix
            y: Training labels
            feature_names: List of feature names
            class_labels: List of class labels
            eval_set: Evaluation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
        
        self.feature_names = feature_names or []
        self.class_labels = class_labels or []
        
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = False
        
        self.model.fit(X, y, **fit_params)
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
    ) -> Tuple['XGBoostClassifier', Dict]:
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
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
            }
        
        logger.info(f"Running hyperparameter search with {cv}-fold CV...")
        
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        grid_search = GridSearchCV(
            base_model,
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
            }
        }
        
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return self, results
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        
        # Map back to feature names
        if self.feature_names:
            mapped_importance = {}
            for feat_key, score in importance.items():
                # XGBoost uses f0, f1, f2... as feature names
                if feat_key.startswith('f') and feat_key[1:].isdigit():
                    idx = int(feat_key[1:])
                    if idx < len(self.feature_names):
                        mapped_importance[self.feature_names[idx]] = score
                else:
                    mapped_importance[feat_key] = score
            importance = mapped_importance
        
        # Normalize and sort
        total = sum(importance.values()) if importance else 1
        importance = {k: v / total for k, v in importance.items()}
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
    def load(cls, path: Path) -> 'XGBoostClassifier':
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
    model = XGBoostClassifier(n_estimators=50, max_depth=3)
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

