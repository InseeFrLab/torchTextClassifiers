"""FastText classifier package.

Provides FastText text classification with PyTorch Lightning integration.
This folder contains 4 main files:
- core.py: Configuration, losses, and factory methods
- tokenizer.py: NGramTokenizer implementation
- model.py: PyTorch model, Lightning module, and dataset
- wrapper.py: High-level wrapper interface
"""

from .core import FastTextConfig, OneVsAllLoss, FastTextFactory
from .tokenizer import NGramTokenizer
from .model import FastTextModel, FastTextModule, FastTextModelDataset
from .wrapper import FastTextWrapper

__all__ = [
    "FastTextConfig",
    "OneVsAllLoss", 
    "FastTextFactory",
    "NGramTokenizer",
    "FastTextModel",
    "FastTextModule", 
    "FastTextModelDataset",
    "FastTextWrapper",
]