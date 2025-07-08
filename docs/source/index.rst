**torchTextClassifiers** Documentation
====================================

Welcome to **torchTextClassifiers**, a unified framework for text classification with PyTorch Lightning integration.

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-1.9%2B-orange.svg
   :target: https://pytorch.org
   :alt: PyTorch Version

.. image:: https://img.shields.io/badge/lightning-1.6%2B-purple.svg
   :target: https://pytorch-lightning.readthedocs.io
   :alt: PyTorch Lightning Version

Overview
--------

**torchTextClassifiers** provides a comprehensive, extensible framework for building and training
text classification models. It offers a unified API that abstracts away the complexity
of different model architectures while providing flexibility for advanced users.

🚀 **Key Features:**

* **Unified API**: Consistent interface across different model types
* **PyTorch Lightning Integration**: Production-ready training with automatic GPU handling
* **Mixed Features Support**: Handle both text and categorical data
* **Extensible Architecture**: Easy to add new model types
* **Explainability**: Add explainability support (depends on classifier types)


Quick Start
-----------

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Create classifier
   classifier = create_fasttext(
       embedding_dim=100,
       sparse=False,
       num_tokens=10000,
       min_count=2,
       min_n=3,
       max_n=6,
       len_word_ngrams=2,
       num_classes=2
   )

   # Prepare data
   X_train = np.array(["positive text", "negative text"])
   y_train = np.array([1, 0])

   # Build and train
   classifier.build(X_train, y_train)
   classifier.train(X_train, y_train, X_train, y_train, num_epochs=10)

   # Predict
   predictions = classifier.predict(np.array(["new text to classify"]))

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   examples
   architecture

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   contributing
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
