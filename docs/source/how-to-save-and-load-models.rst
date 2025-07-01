How to Save and Load Models
===========================

This guide shows you how to properly save and load your trained torchTextClassifiers models 
for production use or to resume training later.

Quick Save/Load
---------------

Save Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import create_fasttext
   import numpy as np

   # Train your model
   classifier = create_fasttext(embedding_dim=100, num_classes=2)
   X_train = np.array(["positive text", "negative text"])
   y_train = np.array([1, 0])

   classifier.build(X_train, y_train)
   classifier.train(X_train, y_train, X_train, y_train, num_epochs=10)

   # Save just the configuration
   classifier.to_json('model_config.json')

Load Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers import torchTextClassifiers

   # Load configuration (creates new instance)
   new_classifier = torchTextClassifiers.from_json('model_config.json')

   # You'll need to rebuild and retrain
   new_classifier.build(X_train, y_train)
   new_classifier.train(X_train, y_train, X_val, y_val, num_epochs=10)

Complete Model Persistence
---------------------------

Save Everything with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import pickle

   # After training your model
   classifier.build(X_train, y_train)
   classifier.train(X_train, y_train, X_val, y_val, num_epochs=20)

   # Method 1: Save the entire classifier object
   torch.save(classifier, 'complete_model.pth')

   # Method 2: Save components separately (recommended)
   model_state = {
       'classifier': classifier,
       'config': classifier.config,
       'tokenizer_state': classifier.tokenizer.__dict__ if hasattr(classifier, 'tokenizer') else None,
       'model_state_dict': classifier.model.state_dict() if hasattr(classifier, 'model') else None
   }
   torch.save(model_state, 'model_components.pth')

Load Complete Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   # Method 1: Load entire classifier
   loaded_classifier = torch.load('complete_model.pth')
   loaded_classifier.eval()  # Set to evaluation mode

   # Method 2: Load components (more robust)
   model_state = torch.load('model_components.pth')
   loaded_classifier = model_state['classifier']
   loaded_classifier.eval()

   # Test the loaded model
   predictions = loaded_classifier.predict(np.array(["test text"]))
   print(f"Predictions: {predictions}")

Production-Ready Saving
------------------------

Save for Production Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import json
   from datetime import datetime

   def save_production_model(classifier, model_name, version="1.0"):
       """Save model with metadata for production use"""
       
       # Create metadata
       metadata = {
           'model_name': model_name,
           'version': version,
           'created_at': datetime.now().isoformat(),
           'config': classifier.config.__dict__,
           'num_classes': classifier.config.num_classes,
           'embedding_dim': classifier.config.embedding_dim
       }
       
       # Save model components
       model_package = {
           'classifier': classifier,
           'metadata': metadata
       }
       
       # Save with versioned filename
       filename = f"{model_name}_v{version}.pth"
       torch.save(model_package, filename)
       
       # Save metadata separately for inspection
       with open(f"{model_name}_v{version}_metadata.json", 'w') as f:
           json.dump(metadata, f, indent=2)
       
       print(f"Model saved as {filename}")
       return filename

   # Usage
   model_file = save_production_model(classifier, "sentiment_classifier", "1.0")

Load Production Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def load_production_model(model_file):
       """Load production model with validation"""
       
       model_package = torch.load(model_file)
       classifier = model_package['classifier']
       metadata = model_package['metadata']
       
       # Set to evaluation mode  
       classifier.eval()
       
       print(f"Loaded {metadata['model_name']} v{metadata['version']}")
       print(f"Created: {metadata['created_at']}")
       print(f"Classes: {metadata['num_classes']}")
       
       return classifier, metadata

   # Usage
   classifier, info = load_production_model("sentiment_classifier_v1.0.pth")

Handling Model Updates
----------------------

Version Control for Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   from pathlib import Path

   class ModelManager:
       def __init__(self, models_dir="./models"):
           self.models_dir = Path(models_dir)
           self.models_dir.mkdir(exist_ok=True)
       
       def save_model(self, classifier, name, version=None):
           """Save model with automatic versioning"""
           if version is None:
               # Auto-increment version
               existing_versions = self._get_versions(name)
               version = max(existing_versions) + 1 if existing_versions else 1
           
           filename = self.models_dir / f"{name}_v{version}.pth"
           torch.save(classifier, filename)
           
           # Create symlink to latest
           latest_link = self.models_dir / f"{name}_latest.pth"
           if latest_link.exists():
               latest_link.unlink()
           latest_link.symlink_to(filename.name)
           
           return filename, version
       
       def load_latest(self, name):
           """Load the latest version of a model"""
           latest_file = self.models_dir / f"{name}_latest.pth"
           if not latest_file.exists():
               raise FileNotFoundError(f"No model found with name: {name}")
           
           return torch.load(latest_file)
       
       def _get_versions(self, name):
           """Get existing version numbers for a model"""
           pattern = f"{name}_v*.pth"
           files = list(self.models_dir.glob(pattern))
           versions = []
           for file in files:
               try:
                   version = int(file.stem.split('_v')[1])
                   versions.append(version)
               except (IndexError, ValueError):
                   continue
           return versions

   # Usage
   manager = ModelManager()

   # Save new version
   file, version = manager.save_model(classifier, "text_classifier")
   print(f"Saved as version {version}")

   # Load latest version
   latest_classifier = manager.load_latest("text_classifier")

Cross-Platform Compatibility
-----------------------------

Save for Different Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   def save_portable_model(classifier, filename):
       """Save model in a portable format"""
       
       # Save with CPU mapping to ensure compatibility
       if torch.cuda.is_available():
           # Move to CPU before saving
           classifier.cpu()
       
       # Save with explicit CPU mapping
       torch.save(classifier, filename, map_location='cpu')
       
       print(f"Model saved as {filename} (CPU-compatible)")

   def load_portable_model(filename, device=None):
       """Load model and move to appropriate device"""
       
       if device is None:
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Load with explicit device mapping
       classifier = torch.load(filename, map_location=device)
       
       print(f"Model loaded on {device}")
       return classifier

   # Usage
   save_portable_model(classifier, "portable_model.pth")
   loaded_classifier = load_portable_model("portable_model.pth")

Backup and Recovery
-------------------

Checkpoint During Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   def train_with_checkpoints(classifier, X_train, y_train, X_val, y_val, 
                             checkpoint_dir="./checkpoints", save_every=5):
       """Train with automatic checkpointing"""
       
       os.makedirs(checkpoint_dir, exist_ok=True)
       
       # Initial build
       classifier.build(X_train, y_train)
       
       # Train with periodic saves
       for epoch in range(1, 51):  # 50 epochs
           # Train for one epoch
           classifier.train(X_train, y_train, X_val, y_val, 
                           num_epochs=1, verbose=False)
           
           # Save checkpoint
           if epoch % save_every == 0:
               checkpoint_file = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
               torch.save(classifier, checkpoint_file)
               print(f"Checkpoint saved at epoch {epoch}")
       
       return classifier

   # Usage
   classifier = train_with_checkpoints(classifier, X_train, y_train, X_val, y_val)

Common Issues and Solutions
---------------------------

Large Model Files
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For very large models, save with compression
   torch.save(classifier, 'model.pth', _use_new_zipfile_serialization=True)

   # Or save components separately to reduce size
   components = {
       'config': classifier.config,
       'tokenizer': classifier.tokenizer,
       'model_weights': classifier.model.state_dict()
   }
   torch.save(components, 'model_components.pth')

Loading Errors
~~~~~~~~~~~~~~

.. code-block:: python

   try:
       classifier = torch.load('model.pth')
   except Exception as e:
       print(f"Loading failed: {e}")
       print("Try loading with map_location='cpu':")
       classifier = torch.load('model.pth', map_location='cpu')

Version Compatibility
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save version information with your model
   import torch
   import torchTextClassifiers
   import sys

   model_info = {
       'classifier': classifier,
       'torch_version': torch.__version__,
       'package_version': torchTextClassifiers.__version__,
       'python_version': sys.version
   }

   torch.save(model_info, 'model_with_versions.pth')

.. note::
   This guide covers the essential patterns for saving and loading your 
   torchTextClassifiers models safely and efficiently.