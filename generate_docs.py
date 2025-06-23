#!/usr/bin/env python3
"""
Documentation Generation Script for torchTextClassifiers

This script generates comprehensive API documentation using Sphinx and serves it on localhost.
It automatically extracts docstrings, creates API reference pages, and includes custom documentation.

Setup:
    # Install documentation dependencies
    uv sync --group docs
    
    # Or add them manually
    uv add --group docs sphinx sphinx-rtd-theme

Usage:
    uv run python generate_docs.py --build     # Generate documentation
    uv run python generate_docs.py --serve     # Serve on localhost:8000
    uv run python generate_docs.py --all       # Build and serve
    uv run python generate_docs.py --clean     # Clean build directory

Features:
- Auto-generated API documentation from docstrings
- Beautiful HTML theme with navigation
- Architecture diagrams and examples
- Search functionality
- Mobile-responsive design
- Copy-to-clipboard code examples
- Responsive design for mobile/desktop
"""

import os
import sys
import shutil
import subprocess
import argparse
import webbrowser
import http.server
import socketserver
from pathlib import Path
from threading import Timer

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Documentation configuration
DOCS_DIR = project_root / "docs"
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "build"
HTML_DIR = BUILD_DIR / "html"
STATIC_DIR = SOURCE_DIR / "_static"
TEMPLATES_DIR = SOURCE_DIR / "_templates"

def setup_sphinx_environment():
    """Set up Sphinx documentation environment"""
    print("🔧 Setting up Sphinx environment...")
    
    # Create documentation directories
    for dir_path in [DOCS_DIR, SOURCE_DIR, BUILD_DIR, STATIC_DIR, TEMPLATES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create Sphinx configuration
    create_sphinx_conf()
    
    # Create main documentation files
    create_index_rst()
    create_api_rst()
    create_architecture_rst()
    create_examples_rst()
    create_installation_rst()
    
    print("✅ Sphinx environment ready!")

def create_sphinx_conf():
    """Create Sphinx configuration file"""
    conf_content = '''# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'torchTextClassifiers'
copyright = '2024, torchTextClassifiers Contributors'
author = 'torchTextClassifiers Contributors'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Extension configuration ------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Auto-summary
autosummary_generate = True

# Todo extension
todo_include_todos = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'lightning': ('https://pytorch-lightning.readthedocs.io/en/stable/', None),
}

# HTML theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_title = "torchTextClassifiers Documentation"
html_short_title = "torchTextClassifiers"
html_logo = None
html_favicon = None

# Add custom CSS and JS
html_css_files = [
    'custom.css',
]

html_js_files = [
    'custom.js',
]
'''
    
    with open(SOURCE_DIR / "conf.py", "w") as f:
        f.write(conf_content)

def create_index_rst():
    """Create main index documentation"""
    index_content = '''torchTextClassifiers Documentation
====================================

Welcome to torchTextClassifiers, a unified framework for text classification with PyTorch Lightning integration.

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

torchTextClassifiers provides a comprehensive, extensible framework for building and training
text classification models. It offers a unified API that abstracts away the complexity
of different model architectures while providing flexibility for advanced users.

🚀 **Key Features:**

* **Unified API**: Consistent interface across different model types
* **PyTorch Lightning Integration**: Production-ready training with automatic GPU handling
* **FastText Implementation**: Fast and efficient text classification
* **Mixed Features Support**: Handle both text and categorical data
* **Extensible Architecture**: Easy to add new model types

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
'''
    
    with open(SOURCE_DIR / "index.rst", "w") as f:
        f.write(index_content)

def create_api_rst():
    """Create API reference documentation"""
    api_content = '''API Reference
=============

This section contains the complete API reference for torchTextClassifiers.

Core Framework
--------------

.. automodule:: torchTextClassifiers
   :members:
   :undoc-members:
   :show-inheritance:

Main Classifier Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchTextClassifiers.torchTextClassifiers.torchTextClassifiers
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: torchTextClassifiers.torchTextClassifiers.ClassifierType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: torchTextClassifiers.torchTextClassifiers.ClassifierFactory
   :members:
   :undoc-members:
   :show-inheritance:

FastText Classifier
-------------------

Configuration and Factory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: torchTextClassifiers.classifiers.fasttext.fasttext
   :members:
   :undoc-members:
   :show-inheritance:

Tokenizer
~~~~~~~~~

.. automodule:: torchTextClassifiers.classifiers.fasttext.tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

Model Components
~~~~~~~~~~~~~~~~

.. automodule:: torchTextClassifiers.classifiers.fasttext.model
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. automodule:: torchTextClassifiers.classifiers.base
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Preprocessing
~~~~~~~~~~~~~

.. automodule:: torchTextClassifiers.utilities.preprocess
   :members:
   :undoc-members:
   :show-inheritance:

Input Validation
~~~~~~~~~~~~~~~~

.. automodule:: torchTextClassifiers.utilities.checkers
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
~~~~~~~~~~~~~~~~

.. automodule:: torchTextClassifiers.utilities.utils
   :members:
   :undoc-members:
   :show-inheritance:
'''
    
    with open(SOURCE_DIR / "api.rst", "w") as f:
        f.write(api_content)

def create_architecture_rst():
    """Create architecture documentation"""
    arch_content = '''Architecture Overview
====================

This section provides a detailed overview of the torchTextClassifiers architecture,
including design principles, component relationships, and extension points.

Framework Design
----------------

torchTextClassifiers follows a modular, plugin-based architecture that separates
concerns and enables easy extension with new classifier types.

.. code-block:: text

    ┌─────────────────────────────────────────────────────┐
    │                User Interface                       │
    │  create_fasttext(), build(), train(), predict()    │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │              torchTextClassifiers                   │
    │         (Main Classifier Interface)                 │
    │                                                     │
    │  ┌─────────────────┐    ┌─────────────────────────┐ │
    │  │ ClassifierType  │    │  ClassifierFactory      │ │
    │  │   (Enum)        │    │    (Registry)           │ │
    │  └─────────────────┘    └─────────────────────────┘ │
    └─────────────────────┬───────────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────────┐
    │           Classifier Implementations                │
    │                                                     │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │              FastText                           │ │
    │  │                                                 │ │
    │  │ ┌─────────────┐ ┌──────────────┐ ┌────────────┐ │ │
    │  │ │   Config    │ │   Wrapper    │ │   Model    │ │ │
    │  │ │             │ │              │ │            │ │ │
    │  │ └─────────────┘ └──────────────┘ └────────────┘ │ │
    │  │                       │                        │ │
    │  │ ┌─────────────┐ ┌──────▼──────┐ ┌────────────┐ │ │
    │  │ │ Tokenizer   │ │  Lightning  │ │  Dataset   │ │ │
    │  │ │             │ │   Module    │ │            │ │ │
    │  │ └─────────────┘ └─────────────┘ └────────────┘ │ │
    │  └─────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────┘

Core Components
---------------

1. **torchTextClassifiers**: Main classifier interface that provides a unified API
2. **ClassifierType**: Enum defining available classifier types  
3. **ClassifierFactory**: Registry for classifier implementations
4. **BaseClassifierWrapper**: Abstract base class for classifier implementations
5. **BaseClassifierConfig**: Configuration base class

FastText Implementation
-----------------------

The FastText classifier demonstrates the framework's capabilities with a complete
implementation including:

Architecture Flow
~~~~~~~~~~~~~~~~~

.. code-block:: text

    Input Text: "Hello world example"
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │              NGramTokenizer                         │
    │  "Hello" → [hel, ell, llo] + [Hello] + [wor, ord,   │
    │  "world" → [rld] + [world] + [exa, xam, amp, mpl,   │
    │  "example" → [ple, ple] + [example]                 │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Token IDs: [234, 567, 123, ...])
    ┌─────────────────────────────────────────────────────┐
    │              Embedding Layer                        │
    │  - Learnable embedding matrix: [vocab_size, emb_dim]│
    │  - Maps token IDs to dense vectors                  │
    │  - Supports sparse embeddings for memory efficiency │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Embeddings: [seq_len, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │              Pooling Layer                          │
    │  - Average pooling across sequence dimension        │
    │  - Result: [batch_size, emb_dim]                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Pooled features: [batch_size, emb_dim])
    ┌─────────────────────────────────────────────────────┐
    │           Classification Head                       │
    │  - Linear layer: [emb_dim] → [num_classes]          │
    │  - No activation (logits output)                    │
    └─────────────────────────────────────────────────────┘
           │
           ▼ (Logits: [batch_size, num_classes])
        Output Predictions

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~

For text input x = [x₁, x₂, ..., xₙ] and categorical features c = [c₁, c₂, ..., cₘ]:

1. **Token Embeddings**: E(x) = [e₁, e₂, ..., eₙ] where eᵢ ∈ ℝᵈ
2. **Text Representation**: h_text = (1/n) ∑ᵢ eᵢ  
3. **Categorical Embeddings**: h_cat = [E_cat₁(c₁), E_cat₂(c₂), ..., E_catₘ(cₘ)]
4. **Combined Representation**: h = [h_text; h_cat] (concatenation)
5. **Output Logits**: y = W·h + b where W ∈ ℝᶜˣᵈ, b ∈ ℝᶜ

Extension Points
----------------

Adding New Classifiers
~~~~~~~~~~~~~~~~~~~~~~~

To add a new classifier type:

1. Create a new classifier type in the ClassifierType enum
2. Implement BaseClassifierWrapper for your classifier
3. Create a configuration class extending BaseClassifierConfig
4. Register your classifier with ClassifierFactory

Example:

.. code-block:: python

   # 1. Add to ClassifierType enum
   class ClassifierType(Enum):
       FASTTEXT = "fasttext"
       BERT = "bert"  # New classifier type

   # 2. Implement wrapper
   class BertWrapper(BaseClassifierWrapper):
       def __init__(self, config: BertConfig):
           super().__init__(config)
           # Implementation...

   # 3. Register with factory
   ClassifierFactory.register_classifier(ClassifierType.BERT, BertWrapper)

Design Principles
-----------------

1. **Separation of Concerns**: Each component has a single responsibility
2. **Dependency Injection**: Components receive dependencies rather than creating them
3. **Configuration-Driven**: Behavior controlled through configuration objects  
4. **Plugin Architecture**: Easy to add new classifier types
5. **PyTorch Lightning Integration**: Leverage battle-tested training infrastructure
6. **Type Safety**: Strong typing throughout the codebase

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

- Sparse embeddings for large vocabularies
- Lazy loading of model components
- Efficient batch processing
- Memory-mapped dataset loading

Training Optimization
~~~~~~~~~~~~~~~~~~~~~

- Automatic mixed precision support
- Multi-GPU training with PyTorch Lightning
- Gradient accumulation for large batches
- Learning rate scheduling and early stopping

Inference Optimization
~~~~~~~~~~~~~~~~~~~~~~

- Model quantization support
- Batch prediction optimization
- CPU/GPU automatic selection
- Caching for repeated predictions
'''
    
    with open(SOURCE_DIR / "architecture.rst", "w") as f:
        f.write(arch_content)

def create_examples_rst():
    """Create examples documentation"""
    examples_content = '''Examples and Tutorials
=====================

This section provides comprehensive examples showing how to use torchTextClassifiers
for various text classification tasks.

Basic Text Classification
-------------------------

Simple Binary Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Create sample data
   X_train = np.array([
       "I love this product! It's amazing.",
       "This is terrible. Worst purchase ever.",
       "Great quality and fast shipping.",
       "Poor customer service and late delivery.",
       "Excellent value for money.",
       "Would not recommend to anyone."
   ])
   y_train = np.array([1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

   X_val = np.array([
       "Pretty good overall experience.",
       "Not satisfied with the quality."
   ])
   y_val = np.array([1, 0])

   # Create and configure classifier
   classifier = create_fasttext(
       embedding_dim=100,
       sparse=False,
       num_tokens=10000,
       min_count=1,
       min_n=3,
       max_n=6,
       len_word_ngrams=2,
       num_classes=2
   )

   # Build model (this creates the tokenizer and model architecture)
   classifier.build(X_train, y_train)

   # Train the model
   classifier.train(
       X_train, y_train,
       X_val, y_val,
       num_epochs=20,
       batch_size=32,
       lr=0.01
   )

   # Make predictions
   test_texts = np.array([
       "This product is fantastic!",
       "I'm disappointed with my purchase."
   ])
   predictions = classifier.predict(test_texts)
   print(f"Predictions: {predictions}")  # [1, 0]

Multi-class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Multi-class example: topic classification
   X_train = np.array([
       "The stock market reached new highs today",           # finance
       "Scientists discover new planet in distant galaxy",   # science  
       "Local team wins championship after overtime",        # sports
       "New smartphone features revolutionary camera",       # technology
       "Interest rates expected to rise next quarter",       # finance
       "Breakthrough in quantum computing achieved",         # science
       "Olympic records broken in swimming events",          # sports
       "AI model shows impressive language capabilities"     # technology
   ])
   y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # 0=finance, 1=science, 2=sports, 3=tech

   # Create multi-class classifier
   classifier = create_fasttext(
       embedding_dim=128,
       sparse=False,
       num_tokens=15000,
       min_count=1,
       min_n=3,
       max_n=6,
       len_word_ngrams=2,
       num_classes=4  # 4 classes
   )

   classifier.build(X_train, y_train)
   classifier.train(X_train, y_train, X_train, y_train, num_epochs=50)

   # Test predictions
   test_texts = np.array([
       "Cryptocurrency prices surge amid market optimism",
       "Mars rover discovers evidence of ancient water"
   ])
   predictions = classifier.predict(test_texts)
   print(f"Predictions: {predictions}")  # Expected: [0, 1]

Mixed Features Classification
-----------------------------

Text + Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Example: Product review classification with metadata
   # Format: [text, category, brand, price_range]
   X_mixed = np.array([
       ["Great smartphone with excellent camera", 0, 1, 2],  # electronics, brand_1, high_price
       ["Poor quality headphones, broke quickly", 0, 0, 0],  # electronics, brand_0, low_price  
       ["Comfortable running shoes for marathons", 1, 2, 1],  # sports, brand_2, medium_price
       ["Stylish jacket but expensive for quality", 2, 1, 2], # clothing, brand_1, high_price
       ["Budget laptop works fine for basic tasks", 0, 0, 0], # electronics, brand_0, low_price
       ["Professional tennis racket, excellent", 1, 3, 2]    # sports, brand_3, high_price
   ])
   y_train = np.array([1, 0, 1, 0, 1, 1])  # 1=positive review, 0=negative

   # Create classifier with categorical features
   classifier = create_fasttext(
       embedding_dim=64,
       sparse=False,
       num_tokens=8000,
       min_count=1,
       min_n=3,
       max_n=6,
       len_word_ngrams=2,
       num_classes=2,
       # Categorical feature configuration
       categorical_vocabulary_sizes=[3, 4, 3],  # category, brand, price_range vocab sizes
       categorical_embedding_dims=8,            # embedding dimension for each cat feature
       num_categorical_features=3               # number of categorical features
   )

   classifier.build(X_mixed, y_train)
   classifier.train(X_mixed, y_train, X_mixed, y_train, num_epochs=30)

   # Predict with mixed features
   test_mixed = np.array([
       ["Amazing smartwatch with fitness tracking", 0, 1, 2],
       ["Cheap shoes fell apart after one week", 1, 0, 0]
   ])
   predictions = classifier.predict(test_mixed)
   print(f"Mixed feature predictions: {predictions}")

Advanced Configuration
----------------------

Custom Configuration Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers.classifiers.fasttext import FastTextConfig, FastTextWrapper
   from torchTextClassifiers import torchTextClassifiers, ClassifierType

   # Create custom configuration
   config = FastTextConfig(
       # Embedding configuration
       embedding_dim=300,
       sparse=True,  # Use sparse embeddings for memory efficiency
       
       # Tokenizer configuration  
       num_tokens=100000,
       min_count=5,      # Higher threshold for vocabulary inclusion
       min_n=3,
       max_n=6,
       len_word_ngrams=3,  # Longer word n-grams
       
       # Model configuration
       num_classes=5,
       direct_bagging=True,
       
       # Training configuration
       learning_rate=0.005,
       
       # Optional: categorical features
       categorical_vocabulary_sizes=None,
       categorical_embedding_dims=None,
       num_categorical_features=None
   )

   # Create classifier with custom config
   classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)

Large Dataset Handling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Configuration for large datasets
   classifier = create_fasttext(
       embedding_dim=200,
       sparse=True,        # Essential for large vocabularies
       num_tokens=500000,  # Large vocabulary
       min_count=10,       # Filter rare tokens
       min_n=3,
       max_n=5,            # Shorter n-grams for speed
       len_word_ngrams=2,
       num_classes=100     # Many classes
   )

   # Training with larger batches and specific optimizations
   classifier.build(X_train_large, y_train_large)
   classifier.train(
       X_train_large, y_train_large,
       X_val_large, y_val_large,
       num_epochs=10,
       batch_size=512,     # Larger batches for efficiency
       lr=0.01,
       num_workers=4,      # Parallel data loading
       patience_train=2    # Early stopping
   )

Model Evaluation and Analysis
-----------------------------

Validation and Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Train model
   classifier.build(X_train, y_train)
   classifier.train(X_train, y_train, X_val, y_val, num_epochs=20)

   # Evaluate on test set
   test_accuracy = classifier.validate(X_test, y_test)
   print(f"Test accuracy: {test_accuracy:.3f}")

   # Get predictions with confidence
   predictions = classifier.predict(X_test)
   
   # For detailed analysis, use sklearn metrics
   from sklearn.metrics import classification_report, confusion_matrix
   
   print("Classification Report:")
   print(classification_report(y_test, predictions))
   
   print("Confusion Matrix:")
   print(confusion_matrix(y_test, predictions))

Prediction Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze predictions on individual samples
   sample_texts = [
       "This product exceeded my expectations",
       "Terrible quality, would not buy again", 
       "Average product, nothing special"
   ]
   
   predictions = classifier.predict(np.array(sample_texts))
   
   for text, pred in zip(sample_texts, predictions):
       print(f"Text: '{text}'")
       print(f"Prediction: {pred}")
       print("---")

Best Practices
--------------

Data Preprocessing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers.utilities.preprocess import clean_text_feature

   # Clean and normalize text before training
   def preprocess_texts(texts):
       # Use built-in cleaning function
       cleaned = [clean_text_feature(text) for text in texts]
       return np.array(cleaned)

   # Apply preprocessing
   X_train_clean = preprocess_texts(X_train_raw)
   X_val_clean = preprocess_texts(X_val_raw)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parameters to tune for better performance
   param_grid = {
       'embedding_dim': [64, 128, 256],
       'min_count': [1, 2, 5],
       'max_n': [5, 6, 7],
       'len_word_ngrams': [1, 2, 3],
       'lr': [0.001, 0.01, 0.1]
   }

   best_score = 0
   best_params = None

   for embedding_dim in param_grid['embedding_dim']:
       for min_count in param_grid['min_count']:
           # Create classifier with current parameters
           classifier = create_fasttext(
               embedding_dim=embedding_dim,
               min_count=min_count,
               # ... other parameters
           )
           
           # Train and evaluate
           classifier.build(X_train, y_train)
           classifier.train(X_train, y_train, X_val, y_val, num_epochs=10)
           score = classifier.validate(X_val, y_val)
           
           if score > best_score:
               best_score = score
               best_params = {'embedding_dim': embedding_dim, 'min_count': min_count}

   print(f"Best parameters: {best_params}")
   print(f"Best score: {best_score:.3f}")

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save trained model
   import torch
   
   # Save the entire classifier
   torch.save(classifier, 'fasttext_classifier.pth')
   
   # Load in production
   loaded_classifier = torch.load('fasttext_classifier.pth')
   loaded_classifier.eval()  # Set to evaluation mode
   
   # Production prediction function
   def predict_text(text):
       with torch.no_grad():
           prediction = loaded_classifier.predict(np.array([text]))
           return prediction[0]

   # Example usage
   result = predict_text("This is a great product!")
   print(f"Prediction: {result}")

Common Issues and Solutions
---------------------------

Memory Issues
~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, use sparse embeddings
   classifier = create_fasttext(
       sparse=True,           # Enable sparse embeddings
       num_tokens=50000,      # Limit vocabulary size
       min_count=10,          # Filter rare tokens
       embedding_dim=100      # Smaller embeddings
   )

Training Convergence
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If model doesn't converge, try:
   classifier.train(
       X_train, y_train, X_val, y_val,
       num_epochs=100,        # More epochs
       lr=0.001,             # Lower learning rate
       patience_train=10,     # More patience
       batch_size=64         # Smaller batches
   )

Out of Vocabulary Words
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # FastText handles OOV words through character n-grams
   # Ensure appropriate n-gram settings:
   classifier = create_fasttext(
       min_n=3,    # Minimum character n-gram length
       max_n=6,    # Maximum character n-gram length
       min_count=1 # Include rare words in training
   )
'''
    
    with open(SOURCE_DIR / "examples.rst", "w") as f:
        f.write(examples_content)

def create_installation_rst():
    """Create installation documentation"""
    install_content = '''Installation Guide
==================

This guide covers the installation of torchTextClassifiers and its dependencies.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large datasets)
- GPU support optional but recommended for large-scale training

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Core dependencies that will be automatically installed:

- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.6.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0

Optional dependencies for enhanced functionality:

- Jupyter >= 1.0.0 (for notebook examples)
- Matplotlib >= 3.3.0 (for visualization)
- Seaborn >= 0.11.0 (for plotting)

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/torch-fastText.git
   cd torch-fastText

   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in development mode
   pip install -e .

   # Or install with uv (faster)
   uv sync

Using pip
~~~~~~~~~

.. code-block:: bash

   pip install torch-fasttext-classifiers

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda install -c your-channel torch-fasttext-classifiers

GPU Support
-----------

For GPU acceleration, install PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Verify GPU installation:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")

Development Installation
------------------------

For development and contributing:

.. code-block:: bash

   # Clone with development dependencies
   git clone https://github.com/your-org/torch-fastText.git
   cd torch-fastText

   # Install with development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests to verify installation
   pytest tests/

Docker Installation
-------------------

Use the provided Docker image for isolated environment:

.. code-block:: bash

   # Build Docker image
   docker build -t torch-fasttext .

   # Run container
   docker run -it --gpus all torch-fasttext

   # Mount local data
   docker run -it --gpus all -v /path/to/data:/data torch-fasttext

Verification
------------

Verify your installation works correctly:

.. code-block:: python

   # Basic import test
   import torchTextClassifiers
   print(f"torchTextClassifiers version: {torchTextClassifiers.__version__}")

   # Create a simple classifier
   from torchTextClassifiers import create_fasttext
   import numpy as np

   classifier = create_fasttext(
       embedding_dim=50,
       sparse=False,
       num_tokens=1000,
       min_count=1,
       min_n=3,
       max_n=6,
       len_word_ngrams=2,
       num_classes=2
   )

   # Test with dummy data
   X = np.array(["positive text", "negative text"])
   y = np.array([1, 0])

   classifier.build(X, y)
   print("✅ Installation verified successfully!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torchTextClassifiers'**

- Ensure you've activated your virtual environment
- Verify installation with ``pip list | grep torch``
- Try reinstalling: ``pip uninstall torch-fasttext-classifiers && pip install torch-fasttext-classifiers``

**CUDA out of memory**

- Reduce batch size: ``batch_size=32`` → ``batch_size=16``
- Use sparse embeddings: ``sparse=True``
- Reduce embedding dimension: ``embedding_dim=100`` → ``embedding_dim=50``

**Slow training on CPU**

- Install PyTorch with GPU support (see GPU Support section)
- Verify GPU is detected: ``torch.cuda.is_available()``
- Use smaller models for CPU training

**Version conflicts**

.. code-block:: bash

   # Create fresh environment
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install torch-fasttext-classifiers

Performance Optimization
-------------------------

For optimal performance:

**CPU Optimization:**

.. code-block:: bash

   # Install optimized PyTorch build
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Set thread count for better CPU utilization
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4

**GPU Optimization:**

.. code-block:: bash

   # Install appropriate CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Verify GPU memory
   nvidia-smi

**Memory Optimization:**

.. code-block:: python

   # Use these settings for large datasets
   classifier = create_fasttext(
       sparse=True,           # Sparse embeddings
       num_tokens=50000,      # Limit vocabulary
       embedding_dim=100,     # Smaller embeddings
       min_count=5           # Filter rare tokens
   )

Environment Variables
---------------------

Useful environment variables for configuration:

.. code-block:: bash

   # PyTorch settings
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

   # Logging
   export TORCH_TEXT_CLASSIFIERS_LOG_LEVEL=INFO

   # Threading
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4

Platform-Specific Notes
-----------------------

**Windows:**

- Use Command Prompt or PowerShell
- Install Microsoft Visual C++ Build Tools if needed
- Consider using conda for dependency management

**macOS:**

- Install Xcode command line tools: ``xcode-select --install``
- Use homebrew for system dependencies: ``brew install python``

**Linux:**

- Install build essentials: ``sudo apt-get install build-essential``
- For GPU support, install NVIDIA drivers and CUDA toolkit

Next Steps
----------

After successful installation:

1. :doc:`examples` - Run example scripts to get familiar with the API
2. :doc:`architecture` - Understand the framework architecture  
3. :doc:`api` - Explore the complete API reference
4. Start with a simple classification task using your own data

Need Help?
----------

If you encounter issues:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed error information
4. Join our community discussions
'''
    
    with open(SOURCE_DIR / "installation.rst", "w") as f:
        f.write(install_content)

def create_custom_css():
    """Create custom CSS for documentation styling"""
    css_content = '''/* Custom CSS for torchTextClassifiers documentation */

/* Improve code block styling */
.highlight {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Architecture diagrams */
.architecture-diagram {
    font-family: monospace;
    background-color: #f6f8fa;
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    white-space: pre;
}

/* API reference styling */
.api-reference .class {
    border-left: 4px solid #2980b9;
    padding-left: 1rem;
    margin: 1rem 0;
}

/* Examples styling */
.example-section {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 2rem 0;
}

/* Performance tips */
.performance-tip {
    background-color: #e8f5e8;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
}

/* Warning boxes */
.warning {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    margin: 1rem 0;
}

/* Navigation improvements */
.wy-nav-content {
    max-width: 1200px;
}

/* Table styling */
table {
    margin: 1rem 0;
    border-collapse: collapse;
    width: 100%;
}

table th, table td {
    border: 1px solid #dee2e6;
    padding: 0.75rem;
    text-align: left;
}

table th {
    background-color: #f8f9fa;
    font-weight: 600;
}

/* Responsive improvements */
@media (max-width: 768px) {
    .architecture-diagram {
        font-size: 0.8rem;
    }
    
    .wy-nav-content {
        margin-left: 0;
    }
}

/* Syntax highlighting improvements */
.highlight .k { color: #0000ff; } /* Keywords */
.highlight .s { color: #008000; } /* Strings */
.highlight .c { color: #808080; } /* Comments */
.highlight .n { color: #000000; } /* Names */

/* Footer styling */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
    color: #6c757d;
    font-size: 0.9rem;
}
'''
    
    with open(STATIC_DIR / "custom.css", "w") as f:
        f.write(css_content)

def create_custom_js():
    """Create custom JavaScript for documentation"""
    js_content = '''// Custom JavaScript for torchTextClassifiers documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Improve navigation
    improveNavigation();
    
    // Add search enhancements
    enhanceSearch();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        
        button.addEventListener('click', function() {
            const code = block.querySelector('code');
            const text = code ? code.textContent : block.textContent;
            
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        button.addEventListener('mouseenter', function() {
            button.style.opacity = '1';
        });
        
        button.addEventListener('mouseleave', function() {
            button.style.opacity = '0.7';
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
}

function improveNavigation() {
    // Add smooth scrolling to anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Highlight current section in navigation
    highlightCurrentSection();
}

function highlightCurrentSection() {
    const sections = document.querySelectorAll('h1, h2, h3');
    const navLinks = document.querySelectorAll('.wy-menu-vertical a');
    
    function updateActiveLink() {
        let current = '';
        
        sections.forEach(function(section) {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                current = section.id;
            }
        });
        
        navLinks.forEach(function(link) {
            link.classList.remove('current');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('current');
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveLink);
    updateActiveLink();
}

function enhanceSearch() {
    // Add keyboard shortcuts for search
    document.addEventListener('keydown', function(e) {
        // Ctrl+K or Cmd+K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput && searchInput === document.activeElement) {
                searchInput.value = '';
                searchInput.blur();
            }
        }
    });
}

// Add version info to footer
function addVersionInfo() {
    const footer = document.querySelector('.rst-footer-buttons');
    if (footer) {
        const versionInfo = document.createElement('div');
        versionInfo.innerHTML = `
            <div class="footer">
                Built with ❤️ by the torchTextClassifiers team | 
                <a href="https://github.com/your-org/torch-fastText">View on GitHub</a>
            </div>
        `;
        footer.parentNode.insertBefore(versionInfo, footer.nextSibling);
    }
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    addVersionInfo();
});
'''
    
    with open(STATIC_DIR / "custom.js", "w") as f:
        f.write(js_content)

def build_docs():
    """Build Sphinx documentation"""
    print("📚 Building documentation...")
    
    # Set up environment
    setup_sphinx_environment()
    
    # Create custom styling
    create_custom_css()
    create_custom_js()
    
    # Build HTML documentation
    try:
        result = subprocess.run([
            "uv", "run", "--group", "docs", 
            "sphinx-build", "-b", "html", "-E",  # -E forces rebuild
            str(SOURCE_DIR), str(HTML_DIR)
        ], capture_output=True, text=True, check=True)
        
        print("✅ Documentation built successfully!")
        print(f"📁 Documentation available at: {HTML_DIR}")
        
        # Print any warnings
        if result.stderr:
            print("⚠️  Build warnings:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print("❌ Documentation build failed!")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Sphinx not found! Please install documentation dependencies:")
        print("uv add --group docs sphinx sphinx-rtd-theme")
        print("or run: uv sync --group docs")
        sys.exit(1)

def serve_docs(port=8000):
    """Serve documentation on localhost"""
    if not HTML_DIR.exists():
        print("❌ Documentation not built yet. Building first...")
        build_docs()
    
    print(f"🌐 Starting documentation server on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Change to HTML directory
    os.chdir(HTML_DIR)
    
    # Open browser automatically
    def open_browser():
        webbrowser.open(f"http://localhost:{port}")
    
    Timer(1.0, open_browser).start()
    
    # Start server
    try:
        with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"python generate_docs.py --serve --port {port + 1}")
        else:
            print(f"❌ Server error: {e}")

def clean_docs():
    """Clean build directory"""
    if BUILD_DIR.exists():
        print("🧹 Cleaning build directory...")
        shutil.rmtree(BUILD_DIR)
        print("✅ Build directory cleaned!")
    else:
        print("ℹ️  Build directory doesn't exist, nothing to clean")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate and serve API documentation for torchTextClassifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python generate_docs.py --build           # Build documentation
  uv run python generate_docs.py --serve           # Serve on localhost:8000
  uv run python generate_docs.py --serve --port 8080  # Serve on custom port
  uv run python generate_docs.py --all             # Build and serve
  uv run python generate_docs.py --clean           # Clean build directory

Setup:
  uv sync --group docs                             # Install doc dependencies
        """
    )
    
    parser.add_argument("--build", action="store_true", help="Build documentation")
    parser.add_argument("--serve", action="store_true", help="Serve documentation on localhost")
    parser.add_argument("--all", action="store_true", help="Build and serve documentation")
    parser.add_argument("--clean", action="store_true", help="Clean build directory")
    parser.add_argument("--port", type=int, default=8000, help="Port for serving (default: 8000)")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.build, args.serve, args.all, args.clean]):
        parser.print_help()
        return
    
    try:
        if args.clean:
            clean_docs()
        
        if args.build:
            build_docs()
        
        if args.all:
            build_docs()
            serve_docs(args.port)
        elif args.serve:
            serve_docs(args.port)
            
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()