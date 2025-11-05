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

"""

from .torchTextClassifiers import (
    ModelConfig as ModelConfig,
)
from .torchTextClassifiers import (
    TrainingConfig as TrainingConfig,
)
from .torchTextClassifiers import (
    torchTextClassifiers as torchTextClassifiers,
)

__all__ = [
    "torchTextClassifiers",
    "ModelConfig",
    "TrainingConfig",
]

__version__ = "1.0.0"
