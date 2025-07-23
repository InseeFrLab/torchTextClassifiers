"""torchTextClassifiers: A unified framework for text classification.

This package provides a generic, extensible framework for building and training
different types of text classifiers. It currently supports FastText classifiers
with a clean API for building, training, and inference.

Key Features:
- Unified API for different classifier types
- Built-in support for FastText classifiers
- PyTorch Lightning integration for training
- Extensible architecture for adding new classifier types
- Support for both text-only and mixed text/categorical features

Quick Start:
    >>> from torchTextClassifiers import create_fasttext
    >>> import numpy as np
    >>> 
    >>> # Create classifier
    >>> classifier = create_fasttext(
    ...     embedding_dim=100,
    ...     sparse=False,
    ...     num_tokens=10000,
    ...     min_count=2,
    ...     min_n=3,
    ...     max_n=6,
    ...     len_word_ngrams=2,
    ...     num_classes=2
    ... )
    >>> 
    >>> # Prepare data
    >>> X_train = np.array(["positive text", "negative text"])
    >>> y_train = np.array([1, 0])
    >>> X_val = np.array(["validation text"])
    >>> y_val = np.array([1])
    >>> 
    >>> # Build and train
    >>> classifier.build(X_train, y_train)
    >>> classifier.train(X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32)
    >>> 
    >>> # Predict
    >>> predictions = classifier.predict(np.array(["new text sample"]))
"""

from .torchTextClassifiers import torchTextClassifiers

# Convenience imports for FastText
try:
    from .classifiers.fasttext.core import FastTextFactory
    
    # Expose FastText convenience methods at package level for easy access
    create_fasttext = FastTextFactory.create_fasttext
    build_fasttext_from_tokenizer = FastTextFactory.build_from_tokenizer
    
except ImportError:
    # FastText module not available - define placeholder functions
    def create_fasttext(*args, **kwargs):
        raise ImportError("FastText module not available")
    
    def build_fasttext_from_tokenizer(*args, **kwargs):
        raise ImportError("FastText module not available")

__all__ = [
    "torchTextClassifiers",
    "create_fasttext",
    "build_fasttext_from_tokenizer",
]

__version__ = "1.0.0"