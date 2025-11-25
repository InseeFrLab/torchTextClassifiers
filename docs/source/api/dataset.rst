Dataset
=======

PyTorch Dataset classes for data loading.

.. currentmodule:: torchTextClassifiers.dataset

TextClassificationDataset
-------------------------

PyTorch Dataset for text classification with optional categorical features.

.. autoclass:: torchTextClassifiers.dataset.dataset.TextClassificationDataset
   :members:
   :undoc-members:
   :show-inheritance:

   **Features:**

   - Support for text data
   - Optional categorical variables
   - Optional labels (for inference)
   - Multilabel support with ragged arrays
   - Integration with tokenizers

Parameters
----------

.. class:: TextClassificationDataset(X_text, y, tokenizer, X_categorical=None)

   :param X_text: Text samples (list or array of strings)
   :type X_text: Union[List[str], np.ndarray]

   :param y: Labels (optional for inference)
   :type y: Optional[Union[List[int], np.ndarray]]

   :param tokenizer: Tokenizer instance
   :type tokenizer: BaseTokenizer

   :param X_categorical: Categorical features (optional)
   :type X_categorical: Optional[np.ndarray]

Example Usage
-------------

Basic Text Dataset
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchTextClassifiers.dataset import TextClassificationDataset
   from torchTextClassifiers.tokenizers import WordPieceTokenizer
   import numpy as np

   # Prepare data
   texts = ["Text sample 1", "Text sample 2", "Text sample 3"]
   labels = [0, 1, 0]

   # Create tokenizer
   tokenizer = WordPieceTokenizer()
   tokenizer.train(texts, vocab_size=1000)

   # Create dataset
   dataset = TextClassificationDataset(
       X_text=texts,
       y=labels,
       tokenizer=tokenizer
   )

   # Use with DataLoader
   from torch.utils.data import DataLoader

   dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

   for batch in dataloader:
       input_ids, labels_batch = batch
       # Train model...

Mixed Features Dataset
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Text data
   texts = ["Sample 1", "Sample 2", "Sample 3"]
   labels = [0, 1, 2]

   # Categorical data (3 samples, 2 categorical variables)
   categorical = np.array([
       [5, 2],   # Sample 1: cat1=5, cat2=2
       [3, 1],   # Sample 2: cat1=3, cat2=1
       [7, 0],   # Sample 3: cat1=7, cat2=0
   ])

   # Create dataset
   dataset = TextClassificationDataset(
       X_text=texts,
       y=labels,
       tokenizer=tokenizer,
       X_categorical=categorical
   )

   # Batch returns: (input_ids, categorical_features, labels)
   for batch in dataloader:
       input_ids, cat_features, labels_batch = batch
       # Train model with mixed features...

Inference Dataset
~~~~~~~~~~~~~~~~~

For inference without labels:

.. code-block:: python

   # Create dataset without labels
   inference_dataset = TextClassificationDataset(
       X_text=test_texts,
       y=None,  # No labels for inference
       tokenizer=tokenizer
   )

   # Batch returns only features (no labels)
   for batch in dataloader:
       input_ids = batch
       # Make predictions...

Multilabel Dataset
~~~~~~~~~~~~~~~~~~

For multilabel classification:

.. code-block:: python

   # Multilabel targets (ragged arrays supported)
   texts = ["Sample 1", "Sample 2", "Sample 3"]
   labels = [
       [0, 1],     # Sample 1 has labels 0 and 1
       [2],        # Sample 2 has only label 2
       [0, 1, 2],  # Sample 3 has all three labels
   ]

   # Create dataset
   dataset = TextClassificationDataset(
       X_text=texts,
       y=labels,
       tokenizer=tokenizer
   )

   # Dataset handles ragged label arrays automatically

DataLoader Integration
----------------------

The dataset integrates seamlessly with PyTorch DataLoader:

.. code-block:: python

   from torch.utils.data import DataLoader

   # Create dataset
   dataset = TextClassificationDataset(X_text, y, tokenizer)

   # Create dataloader
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True  # For GPU training
   )

   # Iterate
   for batch_idx, batch in enumerate(dataloader):
       # Process batch...
       pass

Batch Format
------------

The dataset returns different batch formats depending on configuration:

**Text only:**

.. code-block:: python

   input_ids = batch
   # Shape: (batch_size, seq_len)

**Text + labels:**

.. code-block:: python

   input_ids, labels = batch
   # input_ids shape: (batch_size, seq_len)
   # labels shape: (batch_size,)

**Text + categorical + labels:**

.. code-block:: python

   input_ids, categorical_features, labels = batch
   # input_ids shape: (batch_size, seq_len)
   # categorical_features shape: (batch_size, num_categorical_vars)
   # labels shape: (batch_size,)

Custom Collation
----------------

For advanced use cases, you can provide a custom collate function:

.. code-block:: python

   def custom_collate_fn(batch):
       # Custom batching logic
       ...
       return custom_batch

   dataloader = DataLoader(
       dataset,
       batch_size=32,
       collate_fn=custom_collate_fn
   )

Memory Considerations
---------------------

For large datasets:

**1. Use generators:**

.. code-block:: python

   def text_generator():
       for text in large_text_file:
           yield text.strip()

   X_text = list(text_generator())

**2. Increase num_workers:**

.. code-block:: python

   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=8  # Parallel data loading
   )

**3. Pin memory for GPU:**

.. code-block:: python

   dataloader = DataLoader(
       dataset,
       batch_size=32,
       pin_memory=True  # Faster GPU transfer
   )

See Also
--------

* :doc:`tokenizers` - Tokenizer options
* :doc:`model` - Using datasets with models
* :doc:`wrapper` - High-level API handling datasets automatically
* :doc:`../tutorials/basic_classification` - Dataset usage examples
