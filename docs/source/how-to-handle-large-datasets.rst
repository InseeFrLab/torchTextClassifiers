How to Handle Large Datasets
============================

This guide provides strategies and best practices for working with large datasets 
that don't fit in memory or require special optimization techniques.

Memory-Efficient Configuration
------------------------------

Sparse Embeddings
~~~~~~~~~~~~~~~~~

For large vocabularies, use sparse embeddings to reduce memory usage:

.. code-block:: python

   from torchTextClassifiers import create_fasttext

   # Configuration for large datasets
   classifier = create_fasttext(
       embedding_dim=200,
       sparse=True,         # Essential for large vocabularies
       num_tokens=500000,   # Large vocabulary
       min_count=10,        # Filter rare tokens
       min_n=3,
       max_n=5,             # Shorter n-grams for speed
       len_word_ngrams=2,
       num_classes=2
   )

Vocabulary Management
~~~~~~~~~~~~~~~~~~~~~

Control vocabulary size to manage memory:

.. code-block:: python

   # Aggressive filtering for very large datasets
   classifier = create_fasttext(
       num_tokens=100000,   # Limit vocabulary size
       min_count=50,        # Higher threshold for inclusion
       sparse=True,         # Use sparse embeddings
       embedding_dim=128    # Moderate embedding size
   )

Batch Processing Strategies
---------------------------

Large Batch Training
~~~~~~~~~~~~~~~~~~~~

Use larger batches to improve training efficiency:

.. code-block:: python

   import numpy as np

   # Configure for large batch processing
   classifier.build(X_train, y_train)
   classifier.train(
       X_train, y_train,
       X_val, y_val,
       batch_size=1024,     # Large batches
       num_epochs=20,
       num_workers=8,       # Parallel data loading
       patience_train=3
   )

Chunked Processing
~~~~~~~~~~~~~~~~~~

Process data in chunks when memory is limited:

.. code-block:: python

   def train_in_chunks(classifier, X_large, y_large, chunk_size=10000):
       """Train on large dataset in chunks"""
       
       # Build with a sample first
       sample_size = min(1000, len(X_large))
       classifier.build(X_large[:sample_size], y_large[:sample_size])
       
       # Process in chunks
       n_chunks = len(X_large) // chunk_size + 1
       
       for epoch in range(10):  # Multiple epochs
           print(f"Epoch {epoch + 1}")
           
           # Shuffle data
           indices = np.random.permutation(len(X_large))
           X_shuffled = X_large[indices]
           y_shuffled = y_large[indices]
           
           for chunk_idx in range(n_chunks):
               start_idx = chunk_idx * chunk_size
               end_idx = min((chunk_idx + 1) * chunk_size, len(X_large))
               
               if start_idx >= len(X_large):
                   break
                   
               X_chunk = X_shuffled[start_idx:end_idx]
               y_chunk = y_shuffled[start_idx:end_idx]
               
               # Train on chunk
               classifier.train(
                   X_chunk, y_chunk,
                   X_chunk, y_chunk,  # Use same data for validation
                   num_epochs=1,
                   verbose=False
               )
               
               print(f"  Processed chunk {chunk_idx + 1}/{n_chunks}")
   
   # Usage
   train_in_chunks(classifier, X_large_dataset, y_large_dataset)

Data Loading Optimization
-------------------------

Efficient Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use efficient data formats and structures:

.. code-block:: python

   import numpy as np
   from typing import List, Tuple
   
   class EfficientDataLoader:
       def __init__(self, texts: List[str], labels: List[int], batch_size: int = 256):
           self.texts = np.array(texts, dtype=object)
           self.labels = np.array(labels)
           self.batch_size = batch_size
           self.indices = np.arange(len(texts))
       
       def __len__(self):
           return len(self.texts) // self.batch_size + (1 if len(self.texts) % self.batch_size else 0)
       
       def shuffle(self):
           """Shuffle the dataset"""
           np.random.shuffle(self.indices)
       
       def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
           """Get a specific batch"""
           start_idx = batch_idx * self.batch_size
           end_idx = min((batch_idx + 1) * self.batch_size, len(self.texts))
           
           batch_indices = self.indices[start_idx:end_idx]
           return self.texts[batch_indices], self.labels[batch_indices]
   
   # Usage
   data_loader = EfficientDataLoader(large_text_list, large_label_list, batch_size=512)
   
   # Train batch by batch
   for epoch in range(num_epochs):
       data_loader.shuffle()
       for batch_idx in range(len(data_loader)):
           X_batch, y_batch = data_loader.get_batch(batch_idx)
           # Process batch...

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Leverage multiple CPU cores for data processing:

.. code-block:: python

   from multiprocessing import Pool
   from functools import partial
   import numpy as np

   def preprocess_chunk(chunk_data, tokenizer_params):
       """Preprocess a chunk of text data"""
       texts, labels = chunk_data
       # Apply preprocessing here
       processed_texts = [text.lower().strip() for text in texts]
       return processed_texts, labels

   def parallel_preprocess(texts, labels, n_processes=4, chunk_size=1000):
       """Preprocess large dataset in parallel"""
       
       # Split data into chunks
       chunks = []
       for i in range(0, len(texts), chunk_size):
           chunk_texts = texts[i:i+chunk_size]
           chunk_labels = labels[i:i+chunk_size]
           chunks.append((chunk_texts, chunk_labels))
       
       # Process chunks in parallel
       with Pool(n_processes) as pool:
           preprocess_func = partial(preprocess_chunk, tokenizer_params={})
           results = pool.map(preprocess_func, chunks)
       
       # Combine results
       all_processed_texts = []
       all_processed_labels = []
       for processed_texts, processed_labels in results:
           all_processed_texts.extend(processed_texts)
           all_processed_labels.extend(processed_labels)
       
       return np.array(all_processed_texts), np.array(all_processed_labels)

   # Usage
   X_processed, y_processed = parallel_preprocess(
       large_text_list, large_label_list, n_processes=8
   )

GPU Memory Management
---------------------

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

Use mixed precision to reduce memory usage:

.. code-block:: python

   # Configure trainer for mixed precision
   trainer_params = {
       'precision': 16,  # Use 16-bit precision
       'accelerator': 'gpu',
       'devices': 1,
       'gradient_clip_val': 1.0,
       'accumulate_grad_batches': 4  # Gradient accumulation
   }

   classifier.train(
       X_train, y_train,
       X_val, y_val,
       batch_size=256,  # Can use larger batches with 16-bit
       trainer_params=trainer_params,
       num_epochs=20
   )

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~

Simulate larger batch sizes with gradient accumulation:

.. code-block:: python

   # Effective batch size = batch_size * accumulate_grad_batches
   trainer_params = {
       'accumulate_grad_batches': 8,  # Accumulate gradients over 8 batches
       'gradient_clip_val': 1.0,
       'max_epochs': 20
   }

   classifier.train(
       X_train, y_train,
       X_val, y_val,
       batch_size=128,  # Small actual batch size
       trainer_params=trainer_params  # But effective size is 128 * 8 = 1024
   )

Data Streaming and Generators
-----------------------------

Generator-Based Training
~~~~~~~~~~~~~~~~~~~~~~~~

Use generators to stream data without loading everything into memory:

.. code-block:: python

   def data_generator(file_path, batch_size=256):
       """Generator that yields batches from a large file"""
       texts, labels = [], []
       
       with open(file_path, 'r') as f:
           for line in f:
               # Parse line (assuming format: "label\ttext")
               parts = line.strip().split('\t', 1)
               if len(parts) == 2:
                   label, text = parts
                   texts.append(text)
                   labels.append(int(label))
                   
                   if len(texts) == batch_size:
                       yield np.array(texts), np.array(labels)
                       texts, labels = [], []
       
       # Yield remaining data
       if texts:
           yield np.array(texts), np.array(labels)

   def train_from_generator(classifier, data_gen, validation_data, num_epochs=10):
       """Train using a data generator"""
       
       for epoch in range(num_epochs):
           print(f"Epoch {epoch + 1}/{num_epochs}")
           
           for batch_idx, (X_batch, y_batch) in enumerate(data_gen):
               # Train on this batch
               classifier.train(
                   X_batch, y_batch,
                   validation_data[0], validation_data[1],
                   num_epochs=1,
                   verbose=False
               )
               
               if batch_idx % 100 == 0:
                   print(f"  Processed {batch_idx} batches")

   # Usage
   gen = data_generator('large_dataset.txt', batch_size=512)
   train_from_generator(classifier, gen, (X_val, y_val))

External Storage Integration
----------------------------

Working with HDF5
~~~~~~~~~~~~~~~~~~

Use HDF5 for efficient storage and retrieval of large datasets:

.. code-block:: python

   import h5py
   import numpy as np

   def save_to_hdf5(texts, labels, filename):
       """Save large dataset to HDF5 format"""
       with h5py.File(filename, 'w') as f:
           # Store texts as variable-length strings
           dt = h5py.special_dtype(vlen=str)
           text_dataset = f.create_dataset('texts', (len(texts),), dtype=dt)
           text_dataset[:] = texts
           
           # Store labels as integers
           f.create_dataset('labels', data=labels)

   def load_from_hdf5_batches(filename, batch_size=256):
       """Load data from HDF5 in batches"""
       with h5py.File(filename, 'r') as f:
           texts = f['texts']
           labels = f['labels']
           
           total_samples = len(texts)
           n_batches = total_samples // batch_size + (1 if total_samples % batch_size else 0)
           
           for i in range(n_batches):
               start_idx = i * batch_size
               end_idx = min((i + 1) * batch_size, total_samples)
               
               batch_texts = texts[start_idx:end_idx]
               batch_labels = labels[start_idx:end_idx]
               
               yield np.array(batch_texts), np.array(batch_labels)

   # Usage
   # Save large dataset
   save_to_hdf5(huge_text_array, huge_label_array, 'large_dataset.h5')

   # Train from HDF5
   classifier.build(X_sample, y_sample)  # Build with small sample first
   
   for epoch in range(num_epochs):
       for X_batch, y_batch in load_from_hdf5_batches('large_dataset.h5', batch_size=512):
           classifier.train(X_batch, y_batch, X_val, y_val, num_epochs=1, verbose=False)

Performance Monitoring
----------------------

Memory Usage Tracking
~~~~~~~~~~~~~~~~~~~~~~

Monitor memory usage during training:

.. code-block:: python

   import psutil
   import gc
   import torch

   def get_memory_usage():
       """Get current memory usage statistics"""
       process = psutil.Process()
       mem_info = process.memory_info()
       
       stats = {
           'cpu_memory_mb': mem_info.rss / 1024 / 1024,
           'cpu_memory_percent': process.memory_percent()
       }
       
       if torch.cuda.is_available():
           stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
           stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
       
       return stats

   def train_with_monitoring(classifier, X_train, y_train, X_val, y_val):
       """Train with memory monitoring"""
       
       print("Starting training with memory monitoring...")
       start_stats = get_memory_usage()
       print(f"Initial memory: {start_stats}")
       
       classifier.build(X_train, y_train)
       build_stats = get_memory_usage()
       print(f"After build: {build_stats}")
       
       classifier.train(X_train, y_train, X_val, y_val, num_epochs=20)
       train_stats = get_memory_usage()
       print(f"After training: {train_stats}")
       
       # Clean up
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()

Performance Optimization Tips
-----------------------------

Vocabulary Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimal settings for different dataset sizes
   
   # Small dataset (< 10K samples)
   small_config = {
       'num_tokens': 10000,
       'min_count': 1,
       'embedding_dim': 100,
       'sparse': False
   }
   
   # Medium dataset (10K - 100K samples)
   medium_config = {
       'num_tokens': 50000,
       'min_count': 2,
       'embedding_dim': 128,
       'sparse': False
   }
   
   # Large dataset (100K - 1M samples)
   large_config = {
       'num_tokens': 100000,
       'min_count': 5,
       'embedding_dim': 200,
       'sparse': True
   }
   
   # Very large dataset (> 1M samples)
   xlarge_config = {
       'num_tokens': 200000,
       'min_count': 10,
       'embedding_dim': 256,
       'sparse': True
   }

Hardware Considerations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   def optimize_for_hardware():
       """Configure based on available hardware"""
       
       device_config = {}
       
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.get_device_properties(0).total_memory
           gpu_memory_gb = gpu_memory / 1024**3
           
           if gpu_memory_gb >= 24:  # High-end GPU
               device_config = {
                   'batch_size': 1024,
                   'num_workers': 8,
                   'precision': 16,
                   'embedding_dim': 512
               }
           elif gpu_memory_gb >= 8:  # Mid-range GPU
               device_config = {
                   'batch_size': 512,
                   'num_workers': 4,
                   'precision': 16,
                   'embedding_dim': 256
               }
           else:  # Low-end GPU
               device_config = {
                   'batch_size': 256,
                   'num_workers': 2,
                   'precision': 32,
                   'embedding_dim': 128
               }
       else:  # CPU only
           device_config = {
               'batch_size': 64,
               'num_workers': 2,
               'precision': 32,
               'embedding_dim': 100
           }
       
       return device_config

   # Usage
   hw_config = optimize_for_hardware()
   print(f"Optimized configuration: {hw_config}")

.. warning::
   When working with very large datasets, always monitor memory usage and 
   start with smaller configurations to test before scaling up.

.. tip::
   Use sparse embeddings for vocabularies larger than 100K tokens to 
   significantly reduce memory usage.