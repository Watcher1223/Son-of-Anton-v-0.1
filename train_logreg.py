#!/usr/bin/env python3
"""
Train Logistic Regression Model

Trains a logistic regression classifier on the API schema extraction dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.models.preprocessor import DataPreprocessor
from src.models.logistic_regression import LogisticRegressionClassifier
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
        description="Train Logistic Regression model for schema extraction prediction"
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
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning'
    )
    
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Inverse regularization strength'
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
    logger.info("LOGISTIC REGRESSION TRAINING")
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
    
    # =========================================================================
    # Model Training
    # =========================================================================
    logger.info("\n[2/4] Training model...")
    
    model = LogisticRegressionClassifier(C=args.C)
    
    if args.tune:
        logger.info("Running hyperparameter search...")
        model, tune_results = model.hyperparameter_search(
            X_train, y_train,
            cv=args.cv,
            scoring='f1_macro'
        )
        logger.info(f"Best parameters: {tune_results['best_params']}")
    else:
        model.fit(X_train, y_train, feature_names=feature_names, class_labels=class_labels)
    
    # Cross-validation on training data
    cv_results = model.cross_validate(X_train, y_train, cv=args.cv, scoring='f1_macro')
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    logger.info("\n[3/4] Evaluating model...")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    evaluator = ModelEvaluator(class_labels=class_labels)
    
    # Training metrics
    train_metrics = evaluator.compute_metrics(y_train, y_pred_train, "LogReg_Train")
    
    # Test metrics
    test_metrics = evaluator.compute_metrics(y_test, y_pred_test, "LogReg_Test")
    
    # Classification report
    logger.info("\nTest Set Classification Report:")
    print(evaluator.get_classification_report(y_test, y_pred_test))
    
    # Practical metrics
    practical = compute_practical_metrics(y_test, y_pred_test)
    logger.info(f"\nPractical Impact:")
    logger.info(f"  Correct predictions: {practical['correct_predictions']}/{practical['total_endpoints']}")
    logger.info(f"  API calls saved: {practical['calls_saved']} ({practical['percent_calls_saved']:.1f}%)")
    
    # Feature importance
    importance = model.get_feature_importance()
    logger.info("\nTop 10 Most Important Features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    logger.info("\n[4/4] Saving model and results...")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = args.output_dir / 'logreg_model.pkl'
    model.save(model_path)
    
    # Save confusion matrix plot
    cm_path = args.output_dir / 'logreg_confusion_matrix.png'
    evaluator.plot_confusion_matrix(
        y_test, y_pred_test,
        model_name="Logistic Regression",
        output_path=cm_path
    )
    
    # Save metrics
    results = {
        'model_name': 'Logistic Regression',
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'cv_results': cv_results,
        'practical_metrics': practical,
        'feature_importance': dict(list(importance.items())[:20]),
        'hyperparameters': {
            'C': model.model.C,
            'solver': model.model.solver,
            'class_weight': model.model.class_weight,
        }
    }
    
    if args.tune:
        results['tune_results'] = tune_results
    
    evaluator.save_results(args.output_dir / 'logreg_results.json', {'LogisticRegression': results})
    
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

