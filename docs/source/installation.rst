Installation Guide
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

Optional dependencies for enhanced functionality:


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
   source venv/bin/activate  # On Windows: venv\Scripts\activate

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
