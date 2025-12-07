#!/usr/bin/env python3
"""
Evaluate and Compare All Models

Loads trained models and generates comprehensive comparison reports.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from src.models.preprocessor import DataPreprocessor
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.xgboost_model import XGBoostClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.models.evaluation import ModelEvaluator, compute_practical_metrics
from src.models.explainability import generate_shap_report


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_model(model_path: Path, model_type: str):
    """Load a trained model from file."""
    if model_type == 'logreg':
        return LogisticRegressionClassifier.load(model_path)
    elif model_type == 'xgboost':
        return XGBoostClassifier.load(model_path)
    elif model_type == 'nn':
        return NeuralNetworkClassifier.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare all trained models"
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/final/dataset.csv'),
        help='Path to dataset CSV'
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('models'),
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Directory to save comparison results'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    
    parser.add_argument(
        '--shap',
        action='store_true',
        help='Generate SHAP analysis (can be slow)'
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
    # Setup
    # =========================================================================
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION AND COMPARISON")
    logger.info("=" * 70)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n[1/5] Loading and preprocessing data...")
    
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
    
    logger.info(f"Test samples: {X_test.shape[0]}")
    logger.info(f"Features: {X_test.shape[1]}")
    
    # =========================================================================
    # Load Models
    # =========================================================================
    logger.info("\n[2/5] Loading trained models...")
    
    models: Dict[str, Any] = {}
    model_files = {
        'Logistic Regression': ('logreg_model.pkl', 'logreg'),
        'XGBoost': ('xgboost_model.pkl', 'xgboost'),
        'Neural Network': ('nn_model.pt', 'nn'),
    }
    
    for model_name, (filename, model_type) in model_files.items():
        model_path = args.models_dir / filename
        if model_path.exists():
            try:
                models[model_name] = load_model(model_path, model_type)
                logger.info(f"  Loaded: {model_name}")
            except Exception as e:
                logger.warning(f"  Failed to load {model_name}: {e}")
        else:
            logger.warning(f"  Not found: {model_name} ({model_path})")
    
    if not models:
        logger.error("No models found! Please train models first.")
        logger.error("Run: python train_logreg.py && python train_xgboost.py && python train_nn.py")
        sys.exit(1)
    
    # =========================================================================
    # Evaluate Models
    # =========================================================================
    logger.info("\n[3/5] Evaluating models on test set...")
    
    evaluator = ModelEvaluator(class_labels=class_labels)
    predictions: Dict[str, np.ndarray] = {}
    all_results: Dict[str, Dict] = {}
    
    for model_name, model in models.items():
        logger.info(f"\n--- {model_name} ---")
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        
        # Compute metrics
        metrics = evaluator.compute_metrics(y_test, y_pred, model_name)
        
        # Practical metrics
        practical = compute_practical_metrics(y_test, y_pred)
        
        # Classification report
        logger.info(f"\nClassification Report:")
        print(evaluator.get_classification_report(y_test, y_pred))
        
        all_results[model_name] = {
            'metrics': metrics,
            'practical_metrics': practical,
        }
    
    # =========================================================================
    # Compare Models
    # =========================================================================
    logger.info("\n[4/5] Generating comparison reports...")
    
    # Comparison table
    comparison_df = evaluator.compare_models()
    comparison_path = args.output_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved comparison table to {comparison_path}")
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print()
    
    # Metrics comparison bar chart
    evaluator.plot_metrics_comparison(
        output_path=args.output_dir / 'metrics_comparison.png'
    )
    plt.close()
    
    # Combined confusion matrices
    evaluator.plot_all_confusion_matrices(
        all_results,
        y_test,
        predictions,
        output_path=args.output_dir / 'confusion_matrices.png'
    )
    plt.close()
    
    # Save all results to JSON
    results_json = {
        name: {
            'accuracy': res['metrics']['accuracy'],
            'f1_macro': res['metrics']['f1_macro'],
            'f1_weighted': res['metrics']['f1_weighted'],
            'precision_macro': res['metrics']['precision_macro'],
            'recall_macro': res['metrics']['recall_macro'],
            'confusion_matrix': res['metrics']['confusion_matrix'],
            'practical_metrics': res['practical_metrics'],
        }
        for name, res in all_results.items()
    }
    
    with open(args.output_dir / 'all_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # =========================================================================
    # SHAP Analysis (Optional)
    # =========================================================================
    if args.shap:
        logger.info("\n[5/5] Generating SHAP analysis...")
        
        shap_results = {}
        
        for model_name, model in models.items():
            if model_name == 'Neural Network':
                logger.info(f"  Skipping SHAP for {model_name} (use KernelExplainer if needed)")
                continue
            
            try:
                shap_result = generate_shap_report(
                    model,
                    X_train,
                    X_test,
                    feature_names,
                    class_labels,
                    args.output_dir / 'shap',
                    model_name.replace(' ', '_')
                )
                shap_results[model_name] = shap_result
                logger.info(f"  Generated SHAP for {model_name}")
            except Exception as e:
                logger.warning(f"  SHAP failed for {model_name}: {e}")
        
        # Save SHAP results
        if shap_results:
            shap_json = {
                name: {
                    'top_10_features': [
                        {'feature': f, 'importance': float(v)}
                        for f, v in res['top_10_features']
                    ]
                }
                for name, res in shap_results.items()
            }
            
            with open(args.output_dir / 'shap_results.json', 'w') as f:
                json.dump(shap_json, f, indent=2)
    else:
        logger.info("\n[5/5] Skipping SHAP analysis (use --shap to enable)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['metrics']['f1_macro'])
    
    logger.info(f"\nBest Model: {best_model[0]}")
    logger.info(f"  Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {best_model[1]['metrics']['f1_macro']:.4f}")
    logger.info(f"  API Calls Saved: {best_model[1]['practical_metrics']['percent_calls_saved']:.1f}%")
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("\nGenerated files:")
    for f in args.output_dir.glob('*'):
        if f.is_file():
            logger.info(f"  - {f.name}")
    
    # Print practical impact summary
    print("\n" + "=" * 70)
    print("PRACTICAL IMPACT SUMMARY")
    print("=" * 70)
    
    for model_name, results in all_results.items():
        pm = results['practical_metrics']
        print(f"\n{model_name}:")
        print(f"  Correct predictions: {pm['correct_predictions']}/{pm['total_endpoints']} ({100*pm['correct_predictions']/pm['total_endpoints']:.1f}%)")
        print(f"  API calls without model: {pm['calls_without_model']}")
        print(f"  API calls with model: {pm['calls_with_model']}")
        print(f"  Calls saved: {pm['calls_saved']} ({pm['percent_calls_saved']:.1f}%)")


if __name__ == "__main__":
    main()

