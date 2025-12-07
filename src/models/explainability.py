"""
Model Explainability using SHAP

Provides SHAP-based feature importance analysis for model interpretability.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for schema extraction classifiers.
    
    Provides:
    - Global feature importance
    - Per-class feature importance
    - Local explanations for individual predictions
    - Visualization of SHAP values
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_labels: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (sklearn-compatible or custom)
            feature_names: List of feature names
            class_labels: List of class labels
        """
        self.model = model
        self.feature_names = feature_names
        self.class_labels = class_labels or ['openapi', 'validator', 'typedef']
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.background_data: Optional[np.ndarray] = None
    
    def fit(
        self,
        X_background: np.ndarray,
        max_background_samples: int = 100
    ) -> 'SHAPExplainer':
        """
        Fit SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP
            max_background_samples: Maximum samples to use for background
            
        Returns:
            Self for method chaining
        """
        # Subsample background if too large
        if len(X_background) > max_background_samples:
            indices = np.random.choice(len(X_background), max_background_samples, replace=False)
            X_background = X_background[indices]
        
        self.background_data = X_background
        
        logger.info(f"Fitting SHAP explainer with {len(X_background)} background samples")
        
        # Determine explainer type based on model
        if hasattr(self.model, 'model'):
            # Wrapper class - get underlying model
            underlying_model = self.model.model
        else:
            underlying_model = self.model
        
        # Check model type and create appropriate explainer
        model_type = type(underlying_model).__name__
        
        if 'XGB' in model_type or 'xgb' in model_type.lower():
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(underlying_model)
            logger.info("Using TreeExplainer for XGBoost model")
        elif 'LogisticRegression' in model_type:
            # Use LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(underlying_model, X_background)
            logger.info("Using LinearExplainer for Logistic Regression")
        else:
            # Use KernelExplainer as fallback (works with any model)
            def predict_fn(X):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                else:
                    return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(predict_fn, X_background)
            logger.info("Using KernelExplainer (generic)")
        
        return self
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Feature matrix to explain
            max_samples: Maximum samples to explain (None for all)
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted first")
        
        if max_samples is not None and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def get_global_importance(
        self,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values (uses stored values if None)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        shap_values = shap_values if shap_values is not None else self.shap_values
        
        if shap_values is None:
            raise RuntimeError("SHAP values must be computed first")
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class output: list of arrays per class
            # Average across classes
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Single array
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dict
        importance = dict(zip(self.feature_names, mean_abs_shap))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def get_class_importance(
        self,
        class_idx: int,
        shap_values: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get feature importance for a specific class.
        
        Args:
            class_idx: Index of the class
            shap_values: SHAP values (uses stored values if None)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        shap_values = shap_values if shap_values is not None else self.shap_values
        
        if shap_values is None:
            raise RuntimeError("SHAP values must be computed first")
        
        if isinstance(shap_values, list):
            if class_idx >= len(shap_values):
                raise IndexError(f"Class index {class_idx} out of range")
            class_shap = shap_values[class_idx]
        else:
            class_shap = shap_values
        
        mean_abs_shap = np.abs(class_shap).mean(axis=0)
        
        importance = dict(zip(self.feature_names, mean_abs_shap))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def plot_summary(
        self,
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 20,
        output_path: Optional[Path] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot SHAP summary (beeswarm) plot.
        
        Args:
            X: Feature matrix
            shap_values: SHAP values
            max_display: Maximum features to display
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        shap_values = shap_values if shap_values is not None else self.shap_values
        
        if shap_values is None:
            raise RuntimeError("SHAP values must be computed first")
        
        # Handle multi-class by averaging
        if isinstance(shap_values, list):
            shap_values_plot = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values_plot = shap_values
        
        fig = plt.figure(figsize=figsize)
        
        shap.summary_plot(
            shap_values_plot,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {output_path}")
        
        return fig
    
    def plot_bar(
        self,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 20,
        output_path: Optional[Path] = None,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        """
        Plot SHAP bar chart of feature importance.
        
        Args:
            shap_values: SHAP values
            max_display: Maximum features to display
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        importance = self.get_global_importance(shap_values)
        
        # Get top features
        top_features = list(importance.items())[:max_display]
        features, values = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Feature Importance (SHAP)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP bar plot saved to {output_path}")
        
        return fig
    
    def plot_class_comparison(
        self,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 15,
        output_path: Optional[Path] = None,
        figsize: tuple = (14, 6)
    ) -> plt.Figure:
        """
        Plot feature importance comparison across classes.
        
        Args:
            shap_values: SHAP values
            max_display: Maximum features to display
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        shap_values = shap_values if shap_values is not None else self.shap_values
        
        if shap_values is None:
            raise RuntimeError("SHAP values must be computed first")
        
        if not isinstance(shap_values, list):
            logger.warning("Class comparison requires multi-class SHAP values")
            return plt.figure()
        
        n_classes = min(len(shap_values), len(self.class_labels))
        
        fig, axes = plt.subplots(1, n_classes, figsize=figsize)
        
        for i, (ax, class_label) in enumerate(zip(axes, self.class_labels[:n_classes])):
            importance = self.get_class_importance(i, shap_values)
            top_features = list(importance.items())[:max_display]
            features, values = zip(*top_features)
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=f'C{i}')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title(f'{class_label}')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Feature Importance by Class', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Class comparison plot saved to {output_path}")
        
        return fig
    
    def explain_prediction(
        self,
        X_single: np.ndarray,
        prediction: int
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single sample feature vector
            prediction: Predicted class index
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise RuntimeError("Explainer must be fitted first")
        
        # Ensure 2D input
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        shap_vals = self.explainer.shap_values(X_single)
        
        if isinstance(shap_vals, list):
            sample_shap = shap_vals[prediction][0]
        else:
            sample_shap = shap_vals[0]
        
        # Get top contributing features
        feature_contributions = list(zip(self.feature_names, sample_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = {
            'predicted_class': self.class_labels[prediction],
            'predicted_class_idx': prediction,
            'top_positive_features': [
                (f, v) for f, v in feature_contributions if v > 0
            ][:5],
            'top_negative_features': [
                (f, v) for f, v in feature_contributions if v < 0
            ][:5],
            'all_contributions': dict(feature_contributions),
        }
        
        return explanation


def generate_shap_report(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    class_labels: List[str],
    output_dir: Path,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Generate complete SHAP analysis report.
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data to explain
        feature_names: Feature names
        class_labels: Class labels
        output_dir: Directory for output files
        model_name: Name for the model
        
    Returns:
        Dictionary with importance results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nGenerating SHAP report for {model_name}...")
    
    # Create explainer
    explainer = SHAPExplainer(model, feature_names, class_labels)
    explainer.fit(X_train, max_background_samples=100)
    
    # Compute SHAP values
    shap_values = explainer.compute_shap_values(X_test, max_samples=200)
    
    # Get importance
    global_importance = explainer.get_global_importance()
    
    # Generate plots
    try:
        explainer.plot_bar(
            max_display=20,
            output_path=output_dir / f'{model_name.lower()}_shap_importance.png'
        )
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate bar plot: {e}")
    
    try:
        explainer.plot_summary(
            X_test[:min(200, len(X_test))],
            max_display=20,
            output_path=output_dir / f'{model_name.lower()}_shap_summary.png'
        )
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate summary plot: {e}")
    
    return {
        'global_importance': global_importance,
        'top_10_features': list(global_importance.items())[:10],
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from sklearn.linear_model import LogisticRegression
    
    # Create dummy data and model
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(10)]
    
    # Create explainer
    explainer = SHAPExplainer(model, feature_names)
    explainer.fit(X[:50])
    
    # Compute SHAP values
    shap_values = explainer.compute_shap_values(X[50:])
    
    # Get importance
    importance = explainer.get_global_importance()
    
    print("\nTop 5 features by SHAP importance:")
    for i, (feat, imp) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feat}: {imp:.4f}")

