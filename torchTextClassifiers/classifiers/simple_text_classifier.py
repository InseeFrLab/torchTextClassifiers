"""
Simple text classifier example that doesn't require a tokenizer.

This demonstrates how to create a classifier wrapper that uses 
different text preprocessing approaches.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim import Adam

from .base import BaseClassifierWrapper, BaseClassifierConfig


@dataclass
class SimpleTextConfig(BaseClassifierConfig):
    """Configuration for simple text classifier using TF-IDF."""
    
    hidden_dim: int = 128
    num_classes: Optional[int] = None
    max_features: int = 10000
    learning_rate: float = 1e-3
    dropout_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleTextConfig":
        return cls(**data)


class SimpleTextDataset(Dataset):
    """Dataset for simple text classifier."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleTextModel(nn.Module):
    """Simple neural network for text classification using TF-IDF features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleTextModule(pl.LightningModule):
    """Lightning module for simple text classifier."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)


class SimpleTextWrapper(BaseClassifierWrapper):
    """Wrapper for simple text classifier that uses TF-IDF instead of tokenization."""
    
    def __init__(self, config: SimpleTextConfig):
        super().__init__(config)
        self.config: SimpleTextConfig = config
        self.vectorizer: Optional[TfidfVectorizer] = None
        
    def prepare_text_features(self, training_text: np.ndarray) -> None:
        """Prepare TF-IDF vectorizer instead of tokenizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            lowercase=True,
            stop_words='english'
        )
        # Fit the vectorizer on training text
        self.vectorizer.fit(training_text)
    
    def _build_pytorch_model(self) -> None:
        """Build the PyTorch model."""
        if self.vectorizer is None:
            raise ValueError("Must call prepare_text_features first")
        
        input_dim = len(self.vectorizer.get_feature_names_out())
        
        self.pytorch_model = SimpleTextModel(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        )
    
    def _check_and_init_lightning(self, **kwargs) -> None:
        """Initialize Lightning module."""
        self.lightning_module = SimpleTextModule(
            model=self.pytorch_model,
            learning_rate=self.config.learning_rate
        )
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        if not self.trained:
            raise Exception("Model must be trained first.")
        
        # Extract text from X (assuming first column is text)
        text_data = X[:, 0] if X.ndim > 1 else X
        
        # Transform text to TF-IDF features
        features = self.vectorizer.transform(text_data).toarray()
        features_tensor = torch.FloatTensor(features)
        
        self.pytorch_model.eval()
        with torch.no_grad():
            logits = self.pytorch_model(features_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.numpy()
    
    def validate(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Validate the model."""
        predictions = self.predict(X)
        accuracy = (predictions == Y).mean()
        return float(accuracy)
    
    def create_dataset(self, texts: np.ndarray, labels: np.ndarray, categorical_variables: Optional[np.ndarray] = None):
        """Create dataset."""
        # Transform text to TF-IDF features
        features = self.vectorizer.transform(texts).toarray()
        return SimpleTextDataset(features, labels)
    
    def create_dataloader(self, dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True):
        """Create dataloader."""
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    def load_best_model(self, checkpoint_path: str) -> None:
        """Load best model from checkpoint."""
        self.lightning_module = SimpleTextModule.load_from_checkpoint(
            checkpoint_path,
            model=self.pytorch_model,
            learning_rate=self.config.learning_rate
        )
        self.pytorch_model = self.lightning_module.model
        self.trained = True
        self.pytorch_model.eval()
    
    @classmethod
    def get_config_class(cls):
        """Return the configuration class."""
        return SimpleTextConfig