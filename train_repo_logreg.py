#!/usr/bin/env python3
"""
Train Logistic Regression model for repository-level schema classification.

Uses only repo-level features obtainable via GitHub API (file tree + package.json).
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.repo_preprocessor import RepoPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, class_labels, output_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title='Repo-Level Logistic Regression\nConfusion Matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def train_model(data_path: str, output_dir: str, random_state: int = 42):
    """
    Train logistic regression model on repo-level data.
    
    Args:
        data_path: Path to repo-level dataset CSV
        output_dir: Directory to save model and results
        random_state: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor and prepare data
    logger.info("Loading and preprocessing data...")
    preprocessor = RepoPreprocessor()
    data = preprocessor.prepare_data(Path(data_path), random_state=random_state)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    class_labels = data['class_labels']
    feature_names = data['feature_names']
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Classes: {class_labels}")
    
    # Train logistic regression with class weighting for imbalanced data
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        solver='lbfgs',
        multi_class='multinomial'
    )
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (macro): {precision_macro:.4f}")
    logger.info(f"  Recall (macro): {recall_macro:.4f}")
    logger.info(f"  F1 Score (macro): {f1_macro:.4f}")
    
    # Full classification report
    report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_labels, output_dir / "repo_logreg_confusion_matrix.png")
    
    # Feature importance (coefficients)
    logger.info("\nTop 10 Features by Importance (abs mean coefficient):")
    coef_importance = np.abs(model.coef_).mean(axis=0)
    top_indices = np.argsort(coef_importance)[::-1][:10]
    for i, idx in enumerate(top_indices):
        logger.info(f"  {i+1}. {feature_names[idx]}: {coef_importance[idx]:.4f}")
    
    # Save results
    results = {
        'model_type': 'logistic_regression',
        'level': 'repository',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'num_features': int(X_train.shape[1]),
        'classes': class_labels,
        'metrics': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
        },
        'confusion_matrix': cm.tolist(),
        'feature_names': feature_names,
        'top_features': [
            {'feature': feature_names[idx], 'importance': float(coef_importance[idx])}
            for idx in top_indices
        ],
        'hyperparameters': {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'lbfgs',
            'multi_class': 'multinomial',
        }
    }
    
    results_path = output_dir / "repo_logreg_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save model and preprocessor
    model_path = output_dir / "repo_logreg_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'class_labels': class_labels,
        }, f)
    logger.info(f"Saved model to {model_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train repo-level logistic regression')
    parser.add_argument('--data', type=str, default='data/final/repo_dataset.csv',
                        help='Path to repo-level dataset')
    parser.add_argument('--output', type=str, default='models/repo_level',
                        help='Output directory for model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    results = train_model(args.data, args.output, args.seed)
    
    logger.info("\n" + "="*60)
    logger.info("REPO-LEVEL LOGISTIC REGRESSION TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 (macro): {results['metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

