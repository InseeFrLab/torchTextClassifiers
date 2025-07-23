from typing import Optional, Union, Type, List, Dict, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import numpy as np

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
    """Abstract base class for classifier wrappers.
    
    Each classifier wrapper is responsible for its own text processing approach.
    Some may use tokenizers, others may use different preprocessing methods.
    """
    
    def __init__(self, config: BaseClassifierConfig):
        self.config = config
        self.pytorch_model = None
        self.lightning_module = None
        self.trained: bool = False
        self.device = None
        # Remove tokenizer from base class - it's now wrapper-specific
    
    @abstractmethod
    def prepare_text_features(self, training_text: np.ndarray) -> None:
        """Prepare text features for the classifier.
        
        This could involve tokenization, vectorization, or other preprocessing.
        Each classifier wrapper implements this according to its needs.
        """
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
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[BaseClassifierConfig]:
        """Return the configuration class for this wrapper."""
        pass