#!/usr/bin/env python3
"""
Train XGBoost model for repository-level schema classification.

Uses only repo-level features obtainable via GitHub API (file tree + package.json).
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
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
        title='Repo-Level XGBoost\nConfusion Matrix',
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


def plot_feature_importance(model, feature_names, output_path, top_n=15):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importance[indices][::-1])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Repo-Level XGBoost Feature Importance')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


def train_model(data_path: str, output_dir: str, random_state: int = 42):
    """
    Train XGBoost model on repo-level data.
    
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
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = total / (len(class_counts) * class_counts)
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    # Train XGBoost with hyperparameters tuned for small datasets
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,  # Shallow trees to prevent overfitting on small data
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
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
    plot_confusion_matrix(cm, class_labels, output_dir / "repo_xgboost_confusion_matrix.png")
    
    # Feature importance
    plot_feature_importance(model, feature_names, output_dir / "repo_xgboost_feature_importance.png")
    
    logger.info("\nTop 10 Features by Importance:")
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:10]
    for i, idx in enumerate(top_indices):
        logger.info(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Save results
    results = {
        'model_type': 'xgboost',
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
            {'feature': feature_names[idx], 'importance': float(importance[idx])}
            for idx in top_indices
        ],
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 2,
        }
    }
    
    results_path = output_dir / "repo_xgboost_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save model and preprocessor
    model_path = output_dir / "repo_xgboost_model.pkl"
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
    parser = argparse.ArgumentParser(description='Train repo-level XGBoost model')
    parser.add_argument('--data', type=str, default='data/final/repo_dataset.csv',
                        help='Path to repo-level dataset')
    parser.add_argument('--output', type=str, default='models/repo_level',
                        help='Output directory for model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    results = train_model(args.data, args.output, args.seed)
    
    logger.info("\n" + "="*60)
    logger.info("REPO-LEVEL XGBOOST TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 (macro): {results['metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

