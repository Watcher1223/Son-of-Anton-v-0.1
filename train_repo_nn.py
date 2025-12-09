#!/usr/bin/env python3
"""
Train Neural Network model for repository-level schema classification.

Uses only repo-level features obtainable via GitHub API (file tree + package.json).
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.repo_preprocessor import RepoPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepoClassifierNN(nn.Module):
    """Simple neural network for repo-level classification."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 16]  # Smaller network for small dataset
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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
        title='Repo-Level Neural Network\nConfusion Matrix',
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


def plot_training_history(history, output_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved training history to {output_path}")


def train_model(data_path: str, output_dir: str, random_state: int = 42):
    """
    Train neural network model on repo-level data.
    
    Args:
        data_path: Path to repo-level dataset CSV
        output_dir: Directory to save model and results
        random_state: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Small batch for small data
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts)).to(device)
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = len(class_labels)
    model = RepoClassifierNN(input_dim, num_classes, hidden_dims=[32, 16]).to(device)
    
    logger.info(f"Model architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 200
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 30
    
    logger.info(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_test_tensor).sum().item() / len(y_test)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    logger.info("Evaluating best model...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
    
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
    plot_confusion_matrix(cm, class_labels, output_dir / "repo_nn_confusion_matrix.png")
    
    # Training history
    plot_training_history(history, output_dir / "repo_nn_training_history.png")
    
    # Save results
    results = {
        'model_type': 'neural_network',
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
        'hyperparameters': {
            'hidden_dims': [32, 16],
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 8,
            'epochs_trained': len(history['train_loss']),
            'best_val_acc': float(best_val_acc),
        },
        'training_history': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1]),
        }
    }
    
    results_path = output_dir / "repo_nn_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save model
    model_path = output_dir / "repo_nn_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_dims': [32, 16],
        'feature_names': feature_names,
        'class_labels': class_labels,
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save preprocessor separately
    import pickle
    preprocessor_path = output_dir / "repo_nn_preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train repo-level neural network')
    parser.add_argument('--data', type=str, default='data/final/repo_dataset.csv',
                        help='Path to repo-level dataset')
    parser.add_argument('--output', type=str, default='models/repo_level',
                        help='Output directory for model and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    results = train_model(args.data, args.output, args.seed)
    
    logger.info("\n" + "="*60)
    logger.info("REPO-LEVEL NEURAL NETWORK TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 (macro): {results['metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

