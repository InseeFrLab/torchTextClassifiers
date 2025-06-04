from typing import Optional, Union, Type, List, Dict, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import numpy as np
from .fasttext.tokenizer import NGramTokenizer

class BaseClassifierConfig(ABC):
    """Abstract base class for classifier configurations."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseClassifierConfig":
        """Create configuration from dictionary."""
        pass

class BaseClassifierWrapper(ABC):
    """Abstract base class for classifier wrappers."""
    
    def __init__(self, config: BaseClassifierConfig):
        self.config = config
        self.tokenizer: Optional[NGramTokenizer] = None
        self.pytorch_model = None
        self.lightning_module = None
        self.trained: bool = False
        self.device = None
    
    @abstractmethod
    def build_tokenizer(self, training_text: np.ndarray) -> None:
        """Build tokenizer from training text."""
        pass
    
    @abstractmethod
    def _build_pytorch_model(self) -> None:
        """Build the PyTorch model."""
        pass
    
    @abstractmethod
    def _check_and_init_lightning(self, **kwargs) -> None:
        """Initialize Lightning module."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def validate(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Validate the model."""
        pass
    
    @abstractmethod
    def create_dataset(self, texts: np.ndarray, labels: np.ndarray, categorical_variables: Optional[np.ndarray] = None):
        """Create dataset for training/validation."""
        pass
    
    @abstractmethod
    def create_dataloader(self, dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True):
        """Create dataloader from dataset."""
        pass
    
    @abstractmethod
    def load_best_model(self, checkpoint_path: str) -> None:
        """Load best model from checkpoint."""
        pass