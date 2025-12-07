#!/usr/bin/env python3
"""
Train Neural Network Model

Trains a PyTorch neural network on the API schema extraction dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.models.preprocessor import DataPreprocessor
from src.models.neural_network import NeuralNetworkClassifier, plot_training_history
from src.models.evaluation import ModelEvaluator, compute_practical_metrics


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural Network model for schema extraction prediction"
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/final/dataset.csv'),
        help='Path to dataset CSV'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models'),
        help='Directory to save trained model and results'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    
    parser.add_argument(
        '--hidden-dim1',
        type=int,
        default=128,
        help='First hidden layer size'
    )
    
    parser.add_argument(
        '--hidden-dim2',
        type=int,
        default=64,
        help='Second hidden layer size'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout probability'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, or auto)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    logger.info("=" * 70)
    logger.info("NEURAL NETWORK TRAINING")
    logger.info("=" * 70)
    
    logger.info("\n[1/4] Loading and preprocessing data...")
    
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data(
        args.data,
        test_size=args.test_size,
        random_state=42
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    class_labels = data['class_labels']
    
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Test samples: {X_test.shape[0]}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Split some training data for validation (for early stopping)
    val_size = int(0.15 * len(X_train))
    indices = np.random.permutation(len(X_train))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    
    logger.info(f"Training split: {len(X_train_split)}")
    logger.info(f"Validation split: {len(X_val)}")
    
    # =========================================================================
    # Model Training
    # =========================================================================
    logger.info("\n[2/4] Training model...")
    logger.info(f"Architecture: {X_train.shape[1]} → {args.hidden_dim1} → {args.hidden_dim2} → {len(class_labels)}")
    
    model = NeuralNetworkClassifier(
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device
    )
    
    model.fit(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        feature_names=feature_names,
        class_labels=class_labels,
        verbose=True
    )
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    logger.info("\n[3/4] Evaluating model...")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    evaluator = ModelEvaluator(class_labels=class_labels)
    
    # Training metrics (on full training set)
    train_metrics = evaluator.compute_metrics(y_train, y_pred_train, "NN_Train")
    
    # Test metrics
    test_metrics = evaluator.compute_metrics(y_test, y_pred_test, "NN_Test")
    
    # Classification report
    logger.info("\nTest Set Classification Report:")
    print(evaluator.get_classification_report(y_test, y_pred_test))
    
    # Practical metrics
    practical = compute_practical_metrics(y_test, y_pred_test)
    logger.info(f"\nPractical Impact:")
    logger.info(f"  Correct predictions: {practical['correct_predictions']}/{practical['total_endpoints']}")
    logger.info(f"  API calls saved: {practical['calls_saved']} ({practical['percent_calls_saved']:.1f}%)")
    
    # Training history
    history = model.get_training_history()
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    logger.info(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    logger.info("\n[4/4] Saving model and results...")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = args.output_dir / 'nn_model.pt'
    model.save(model_path)
    
    # Save training history plot
    history_path = args.output_dir / 'nn_training_history.png'
    plot_training_history(history, output_path=history_path)
    
    # Save confusion matrix plot
    cm_path = args.output_dir / 'nn_confusion_matrix.png'
    evaluator.plot_confusion_matrix(
        y_test, y_pred_test,
        model_name="Neural Network",
        output_path=cm_path
    )
    
    # Save metrics
    results = {
        'model_name': 'Neural Network',
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'practical_metrics': practical,
        'training_history': {
            'epochs_trained': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
        },
        'hyperparameters': {
            'hidden_dim1': args.hidden_dim1,
            'hidden_dim2': args.hidden_dim2,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_epochs': args.epochs,
            'early_stopping_patience': args.patience,
        }
    }
    
    evaluator.save_results(args.output_dir / 'nn_results.json', {'NeuralNetwork': results})
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"\nTest Performance:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()

