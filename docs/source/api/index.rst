API Reference
=============

Complete API documentation for torchTextClassifiers, auto-generated from source code docstrings.

Overview
--------

The API is organized into several modules:

* :doc:`wrapper` - High-level torchTextClassifiers wrapper class
* :doc:`configs` - Configuration classes (ModelConfig, TrainingConfig)
* :doc:`tokenizers` - Text tokenization (NGram, WordPiece, HuggingFace)
* :doc:`components` - Model components (TextEmbedder, CategoricalVariableNet, etc.)
* :doc:`model` - Core PyTorch models
* :doc:`dataset` - Dataset classes for data loading

Quick Links
-----------

Most Used Classes
~~~~~~~~~~~~~~~~~

* :class:`torchTextClassifiers.torchTextClassifiers.torchTextClassifiers` - Main wrapper class
* :class:`torchTextClassifiers.torchTextClassifiers.ModelConfig` - Model configuration
* :class:`torchTextClassifiers.torchTextClassifiers.TrainingConfig` - Training configuration
* :class:`torchTextClassifiers.tokenizers.WordPieceTokenizer` - WordPiece tokenizer
* :class:`torchTextClassifiers.tokenizers.NGramTokenizer` - N-gram tokenizer

Architecture Components
~~~~~~~~~~~~~~~~~~~~~~~

* :class:`torchTextClassifiers.model.components.TextEmbedder` - Text embedding layer
* :class:`torchTextClassifiers.model.components.CategoricalVariableNet` - Categorical features
* :class:`torchTextClassifiers.model.components.ClassificationHead` - Classification layer
* :class:`torchTextClassifiers.model.components.Attention.AttentionConfig` - Attention configuration

Core Models
~~~~~~~~~~~

* :class:`torchTextClassifiers.model.model.TextClassificationModel` - Core PyTorch model
* :class:`torchTextClassifiers.model.lightning.TextClassificationModule` - PyTorch Lightning module
* :class:`torchTextClassifiers.dataset.dataset.TextClassificationDataset` - PyTorch Dataset

Module Documentation
--------------------

.. toctree::
   :maxdepth: 2

   wrapper
   configs
   tokenizers
   components
   model
   dataset

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
