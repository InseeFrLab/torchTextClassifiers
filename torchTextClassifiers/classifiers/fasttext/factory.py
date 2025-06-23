"""FastText classifier factory and convenience methods.

This module provides factory methods for creating FastText classifiers with
simplified APIs. It offers both high-level convenience methods and advanced
methods for creating classifiers from existing tokenizers.
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

from .config import FastTextConfig
from .wrapper import FastTextWrapper
from .tokenizer import NGramTokenizer

if TYPE_CHECKING:
    from ...torchTextClassifiers import torchTextClassifiers, ClassifierType


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
            >>> from torchTextClassifiers.classifiers.fasttext.factory import FastTextFactory
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
        from ...torchTextClassifiers import torchTextClassifiers, ClassifierType
        
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
        return torchTextClassifiers(ClassifierType.FASTTEXT, config)
    
    @staticmethod
    def build_from_tokenizer(
        tokenizer: NGramTokenizer,
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
        from ...torchTextClassifiers import torchTextClassifiers, ClassifierType
        
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
        
        classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
        classifier.classifier_wrapper.tokenizer = tokenizer
        classifier.classifier_wrapper._build_pytorch_model()
        
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