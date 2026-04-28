"""contrib: example custom architectures for torchTextClassifiers.

These classes are reference implementations that demonstrate how to build custom
PyTorch models compatible with the ``torchTextClassifiers.from_model`` entry point.
They are not part of the core API and may evolve independently.
"""

from .multilevel import MultiLevelCrossEntropyLoss, MultiLevelTextClassificationModel

__all__ = [
    "MultiLevelTextClassificationModel",
    "MultiLevelCrossEntropyLoss",
]
