"""FastText classifier core components.

This module contains the core components for FastText classification:
- Configuration dataclass
- Loss functions
- Factory methods for creating classifiers

Consolidates what was previously in config.py, losses.py, and factory.py.
"""

from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from ..base import BaseClassifierConfig
from typing import Optional, List, TYPE_CHECKING, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ...torchTextClassifiers import torchTextClassifiers


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FastTextConfig(BaseClassifierConfig):
    """Configuration for FastText classifier."""
    # Embedding matrix
    embedding_dim: int
    sparse: bool

    # Tokenizer-related
    num_tokens: int
    min_count: int
    min_n: int
    max_n: int
    len_word_ngrams: int

    # Optional parameters
    num_classes: Optional[int] = None
    num_rows: Optional[int] = None

    # Categorical variables
    categorical_vocabulary_sizes: Optional[List[int]] = None
    categorical_embedding_dims: Optional[Union[List[int], int]] = None
    num_categorical_features: Optional[int] = None

    # Model-specific parameters
    direct_bagging: Optional[bool] = True
    
    # Training parameters
    learning_rate: float = 4e-3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastTextConfig":
        return cls(**data)


# ============================================================================
# Loss Functions
# ============================================================================

class OneVsAllLoss(nn.Module):
    def __init__(self):
        super(OneVsAllLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Compute One-vs-All loss

        Args:
            logits: Tensor of shape (batch_size, num_classes) containing classification scores
            targets: Tensor of shape (batch_size) containing true class indices

        Returns:
            loss: Mean loss value across the batch
        """

        num_classes = logits.size(1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # For each sample, treat the true class as positive and all others as negative
        # Using binary cross entropy for each class
        loss = F.binary_cross_entropy_with_logits(
            logits,  # Raw logits
            targets_one_hot,  # Target probabilities
            reduction="none",  # Don't reduce yet to allow for custom weighting if needed
        )

        # Sum losses across all classes for each sample, then take mean across batch
        return loss.sum(dim=1).mean()


# ============================================================================
# Factory Methods
# ============================================================================

class FastTextFactory:
    """Factory class for creating FastText classifiers with convenience methods.
    
    This factory provides static methods for creating FastText classifiers with
    common configurations. It handles the complexities of configuration creation
    and classifier initialization, offering a simplified API for users.
    
    All methods return fully initialized torchTextClassifiers instances that are
    ready for building and training.
    """
    
    @staticmethod
    def create_fasttext(
        embedding_dim: int,
        sparse: bool,
        num_tokens: int,
        min_count: int,
        min_n: int,
        max_n: int,
        len_word_ngrams: int,
        **kwargs
    ) -> "torchTextClassifiers":
        """Create a FastText classifier with the specified configuration.
        
        This is the primary method for creating FastText classifiers. It creates
        a configuration object with the provided parameters and initializes a
        complete classifier instance.
        
        Args:
            embedding_dim: Dimension of word embeddings
            sparse: Whether to use sparse embeddings
            num_tokens: Maximum number of tokens in vocabulary
            min_count: Minimum count for tokens to be included in vocabulary
            min_n: Minimum length of character n-grams
            max_n: Maximum length of character n-grams
            len_word_ngrams: Length of word n-grams to use
            **kwargs: Additional configuration parameters (e.g., num_classes,
                     categorical_vocabulary_sizes, etc.)
                     
        Returns:
            torchTextClassifiers: Initialized FastText classifier instance
            
        Example:
            >>> from torchTextClassifiers.classifiers.fasttext.core import FastTextFactory
            >>> classifier = FastTextFactory.create_fasttext(
            ...     embedding_dim=100,
            ...     sparse=False,
            ...     num_tokens=10000,
            ...     min_count=2,
            ...     min_n=3,
            ...     max_n=6,
            ...     len_word_ngrams=2,
            ...     num_classes=3
            ... )
        """
        from ...torchTextClassifiers import torchTextClassifiers
        from .wrapper import FastTextWrapper
        
        config = FastTextConfig(
            embedding_dim=embedding_dim,
            sparse=sparse,
            num_tokens=num_tokens,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            **kwargs
        )
        wrapper = FastTextWrapper(config)
        return torchTextClassifiers(wrapper)
    
    @staticmethod
    def build_from_tokenizer(
        tokenizer,  # NGramTokenizer
        embedding_dim: int,
        num_classes: Optional[int],
        categorical_vocabulary_sizes: Optional[List[int]] = None,
        sparse: bool = False,
        **kwargs
    ) -> "torchTextClassifiers":
        """Create FastText classifier from an existing trained tokenizer.
        
        This method is useful when you have a pre-trained tokenizer and want to
        create a classifier that uses the same vocabulary and tokenization scheme.
        The resulting classifier will have its tokenizer and model architecture
        pre-built.
        
        Args:
            tokenizer: Pre-trained NGramTokenizer instance
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            categorical_vocabulary_sizes: Sizes of categorical feature vocabularies
            sparse: Whether to use sparse embeddings
            **kwargs: Additional configuration parameters
            
        Returns:
            torchTextClassifiers: Classifier with pre-built tokenizer and model
            
        Raises:
            ValueError: If the tokenizer is missing required attributes
            
        Example:
            >>> # Assume you have a pre-trained tokenizer
            >>> classifier = FastTextFactory.build_from_tokenizer(
            ...     tokenizer=my_tokenizer,
            ...     embedding_dim=100,
            ...     num_classes=2,
            ...     sparse=False
            ... )
            >>> # The classifier is ready for training without building
            >>> classifier.train(X_train, y_train, X_val, y_val, ...)
        """
        from ...torchTextClassifiers import torchTextClassifiers
        from .wrapper import FastTextWrapper
        
        # Ensure the tokenizer has required attributes
        required_attrs = ["min_count", "min_n", "max_n", "num_tokens", "word_ngrams"]
        if not all(hasattr(tokenizer, attr) for attr in required_attrs):
            missing_attrs = [attr for attr in required_attrs if not hasattr(tokenizer, attr)]
            raise ValueError(f"Missing attributes in tokenizer: {missing_attrs}")
        
        config = FastTextConfig(
            num_tokens=tokenizer.num_tokens,
            embedding_dim=embedding_dim,
            min_count=tokenizer.min_count,
            min_n=tokenizer.min_n,
            max_n=tokenizer.max_n,
            len_word_ngrams=tokenizer.word_ngrams,
            sparse=sparse,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocabulary_sizes,
            **kwargs
        )
        
        wrapper = FastTextWrapper(config)
        classifier = torchTextClassifiers(wrapper)
        classifier.classifier.tokenizer = tokenizer
        classifier.classifier._build_pytorch_model()
        
        return classifier
    
    @staticmethod
    def from_dict(config_dict: dict) -> FastTextConfig:
        """Create FastText configuration from dictionary.
        
        This method is used internally by the configuration factory system
        to recreate FastText configurations from serialized data.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            FastTextConfig: Reconstructed configuration object
            
        Example:
            >>> config_dict = {
            ...     'embedding_dim': 100,
            ...     'num_tokens': 5000,
            ...     'min_count': 1,
            ...     # ... other parameters
            ... }
            >>> config = FastTextFactory.from_dict(config_dict)
        """
        return FastTextConfig.from_dict(config_dict)