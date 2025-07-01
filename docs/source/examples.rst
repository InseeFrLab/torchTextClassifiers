Examples and Tutorials
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
