"""
Neural Network Model for Schema Extraction Prediction

PyTorch 2-layer feedforward network with dropout.
Architecture: Input → 128 → ReLU → Dropout → 64 → ReLU → 3 (softmax)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class SchemaNet(nn.Module):
    """
    2-layer feedforward neural network for schema classification.
    
    Architecture:
        Input → Linear(128) → ReLU → Dropout → Linear(64) → ReLU → Linear(3)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dim1: First hidden layer size
            hidden_dim2: Second hidden layer size
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class NeuralNetworkClassifier:
    """
    Neural Network classifier wrapper for schema extraction prediction.
    
    Provides consistent interface for training, prediction, and model persistence.
    """
    
    def __init__(
        self,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Neural Network classifier.
        
        Args:
            hidden_dim1: First hidden layer size
            hidden_dim2: Second hidden layer size
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Epochs to wait for improvement
            device: Device to use ('cpu', 'cuda', or None for auto)
            config: Additional configuration
        """
        self.config = config or {}
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model: Optional[SchemaNet] = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.class_labels: List[str] = []
        self.input_dim: int = 0
        self.num_classes: int = 3
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'NeuralNetworkClassifier':
        """
        Train the neural network.
        
        Args:
            X: Training feature matrix
            y: Training labels
            X_val: Validation features (optional, enables early stopping)
            y_val: Validation labels
            feature_names: List of feature names
            class_labels: List of class labels
            verbose: Whether to print progress
            
        Returns:
            Self for method chaining
        """
        self.input_dim = X.shape[1]
        self.num_classes = len(np.unique(y))
        self.feature_names = feature_names or []
        self.class_labels = class_labels or []
        
        logger.info(f"Training Neural Network on {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Device: {self.device}")
        
        # Create model
        self.model = SchemaNet(
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create data loaders
        train_loader = self._create_dataloader(X, y, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * batch_X.size(0)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_loss /= val_total
                val_acc = val_correct / val_total
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    )
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with val_loss: {best_val_loss:.4f}")
        
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
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probability matrix
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history.copy()
    
    def save(self, path: Path) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim1': self.hidden_dim1,
            'hidden_dim2': self.hidden_dim2,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'feature_names': self.feature_names,
            'class_labels': self.class_labels,
            'history': self.history,
        }
        
        torch.save(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'NeuralNetworkClassifier':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load model to
            
        Returns:
            Loaded classifier instance
        """
        path = Path(path)
        model_data = torch.load(path, map_location='cpu')
        
        instance = cls(
            hidden_dim1=model_data['hidden_dim1'],
            hidden_dim2=model_data['hidden_dim2'],
            dropout=model_data['dropout'],
            device=device
        )
        
        instance.input_dim = model_data['input_dim']
        instance.num_classes = model_data['num_classes']
        instance.feature_names = model_data['feature_names']
        instance.class_labels = model_data['class_labels']
        instance.history = model_data['history']
        
        # Create and load model
        instance.model = SchemaNet(
            input_dim=instance.input_dim,
            hidden_dim1=instance.hidden_dim1,
            hidden_dim2=instance.hidden_dim2,
            num_classes=instance.num_classes,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(model_data['model_state_dict'])
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        
        return instance


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Path] = None
):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train')
    if history['val_acc']:
        axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(80, 20).astype(np.float32)
    y_train = np.random.randint(0, 3, 80)
    X_val = np.random.randn(20, 20).astype(np.float32)
    y_val = np.random.randint(0, 3, 20)
    
    # Train model
    model = NeuralNetworkClassifier(
        hidden_dim1=64,
        hidden_dim2=32,
        epochs=50,
        early_stopping_patience=10
    )
    
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        feature_names=[f'feat_{i}' for i in range(20)]
    )
    
    # Predict
    y_pred = model.predict(X_val)
    print(f"Predictions shape: {y_pred.shape}")
    
    # Accuracy
    acc = (y_pred == y_val).mean()
    print(f"Validation accuracy: {acc:.4f}")

