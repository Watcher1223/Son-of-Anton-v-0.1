"""
Model Evaluation Utilities

Provides metrics computation, confusion matrix generation, and performance comparison.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates ML model performance with comprehensive metrics.
    
    Computes:
    - Accuracy, precision, recall, F1 (macro and per-class)
    - Confusion matrices
    - Classification reports
    - Model comparison summaries
    """
    
    def __init__(self, class_labels: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            class_labels: List of class names in order
        """
        self.class_labels = class_labels or ['openapi', 'validator', 'typedef']
        self.results: Dict[str, Dict] = {}
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics for a model.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name identifier for the model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(self.class_labels):
            if i < len(precision_per_class):
                metrics[f'precision_{label}'] = float(precision_per_class[i])
                metrics[f'recall_{label}'] = float(recall_per_class[i])
                metrics[f'f1_{label}'] = float(f1_per_class[i])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Store results
        self.results[model_name] = metrics
        
        logger.info(f"\n{model_name} Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report string
        """
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Filter labels and target_names to only include present classes
        labels = [i for i in range(len(self.class_labels)) if i in unique_classes]
        target_names = [self.class_labels[i] for i in labels]
        
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        output_path: Optional[Path] = None,
        normalize: bool = True,
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Model name for title
            output_path: Path to save figure (optional)
            normalize: Whether to normalize values
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        labels = [i for i in range(len(self.class_labels)) if i in unique_classes]
        display_labels = [self.class_labels[i] for i in labels]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {output_path}")
        
        return fig
    
    def plot_all_confusion_matrices(
        self,
        results: Dict[str, Dict],
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        figsize: tuple = (15, 4)
    ) -> plt.Figure:
        """
        Plot confusion matrices for multiple models side by side.
        
        Args:
            results: Dictionary of model results
            y_true: Ground truth labels
            predictions: Dictionary of model predictions
            output_path: Path to save figure
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        # Get unique classes from all predictions combined with y_true
        all_preds = np.concatenate([y_true] + list(predictions.values()))
        unique_classes = np.unique(all_preds)
        labels = [i for i in range(len(self.class_labels)) if i in unique_classes]
        display_labels = [self.class_labels[i] for i in labels]
        
        for ax, (model_name, y_pred) in zip(axes, predictions.items()):
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels,
                ax=ax
            )
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_name}')
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison confusion matrices to {output_path}")
        
        return fig
    
    def compare_models(
        self,
        results: Optional[Dict[str, Dict]] = None,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models.
        
        Args:
            results: Dictionary of model results (uses stored results if None)
            output_path: Path to save comparison CSV
            
        Returns:
            DataFrame with model comparison
        """
        results = results or self.results
        
        if not results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        # Extract key metrics
        comparison_data = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'F1 (Macro)': metrics.get('f1_macro', 0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0),
                'Precision (Macro)': metrics.get('precision_macro', 0),
                'Recall (Macro)': metrics.get('recall_macro', 0),
            }
            
            # Add per-class F1
            for label in self.class_labels:
                row[f'F1 ({label})'] = metrics.get(f'f1_{label}', 0)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 (Macro)', ascending=False)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved model comparison to {output_path}")
        
        return df
    
    def plot_metrics_comparison(
        self,
        results: Optional[Dict[str, Dict]] = None,
        output_path: Optional[Path] = None,
        figsize: tuple = (12, 6)
    ) -> plt.Figure:
        """
        Plot bar chart comparing model metrics.
        
        Args:
            results: Dictionary of model results
            output_path: Path to save figure
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        
        if not results:
            logger.warning("No results to plot")
            return plt.figure()
        
        models = list(results.keys())
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        metric_labels = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            values = [results[m].get(metric, 0) for m in models]
            ax.bar(x + i * width, values, width, label=label)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics comparison plot to {output_path}")
        
        return fig
    
    def save_results(
        self,
        output_path: Path,
        results: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Save all results to JSON file.
        
        Args:
            output_path: Path to save JSON
            results: Dictionary of results (uses stored results if None)
        """
        results = results or self.results
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def load_results(self, input_path: Path) -> Dict[str, Dict]:
        """
        Load results from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Dictionary of results
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.results.update(results)
        logger.info(f"Loaded results from {input_path}")
        
        return results


def compute_practical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    avg_calls_without_prediction: int = 20,
    calls_per_correct: int = 1,
    calls_per_incorrect: int = 15
) -> Dict[str, float]:
    """
    Compute practical metrics: API calls saved by using the model.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        avg_calls_without_prediction: Average API calls without using model
        calls_per_correct: Calls needed when prediction is correct
        calls_per_incorrect: Calls needed when prediction is wrong
        
        
    Returns:
        Dictionary with practical metrics
    """
    n_samples = len(y_true)
    n_correct = np.sum(y_true == y_pred)
    n_incorrect = n_samples - n_correct
    
    # Calculate calls
    calls_without_model = n_samples * avg_calls_without_prediction
    calls_with_model = (n_correct * calls_per_correct) + (n_incorrect * calls_per_incorrect)
    
    calls_saved = calls_without_model - calls_with_model
    percent_saved = (calls_saved / calls_without_model) * 100 if calls_without_model > 0 else 0
    
    return {
        'total_endpoints': int(n_samples),
        'correct_predictions': int(n_correct),
        'incorrect_predictions': int(n_incorrect),
        'calls_without_model': int(calls_without_model),
        'calls_with_model': int(calls_with_model),
        'calls_saved': int(calls_saved),
        'percent_calls_saved': float(percent_saved),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 20, replace=False)] = np.random.randint(0, 3, 20)
    
    evaluator = ModelEvaluator()
    
    # Compute metrics
    metrics = evaluator.compute_metrics(y_true, y_pred, "Example Model")
    
    print("\nClassification Report:")
    print(evaluator.get_classification_report(y_true, y_pred))
    
    # Practical metrics
    practical = compute_practical_metrics(y_true, y_pred)
    print(f"\nPractical Metrics:")
    print(f"  API calls saved: {practical['calls_saved']} ({practical['percent_calls_saved']:.1f}%)")

