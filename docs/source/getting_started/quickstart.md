# Quick Start

This guide will walk you through building your first text classifier with torchTextClassifiers in just a few minutes.

## Overview

In this quick start, you'll:

1. Create sample training data
2. Train a tokenizer
3. Configure a model
4. Train the classifier
5. Make predictions

## Complete Example

Here's a complete, runnable example for sentiment analysis:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For Mac users

from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Step 1: Prepare training data
texts = [
    "I love this product! It's amazing!",
    "Terrible experience, would not recommend.",
    "Pretty good, meets expectations.",
    "Awful quality, very disappointed.",
    "Excellent service and great value!",
    "Not worth the money.",
    "Fantastic! Exceeded my expectations!",
    "Poor quality, broke after one use.",
    "Highly recommend, very satisfied!",
    "Waste of money, terrible product.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Step 2: Create and train tokenizer
print("Training tokenizer...")
tokenizer = WordPieceTokenizer()
tokenizer.train(texts, vocab_size=500, min_frequency=1)
print(f"Tokenizer trained with vocabulary size: {len(tokenizer)}")

# Step 3: Configure model
model_config = ModelConfig(
    embedding_dim=64,  # Size of text embeddings
    num_classes=2,     # Binary classification
)

# Step 4: Configure training
training_config = TrainingConfig(
    num_epochs=10,
    batch_size=4,
    lr=1e-3,
    patience_early_stopping=5,
    accelerator="cpu",  # Use "gpu" if available
)

# Step 5: Create classifier
print("\nCreating classifier...")
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

# Step 6: Train the model
print("\nTraining model...")
classifier.train(
    X_text=texts,
    y=labels,
    training_config=training_config,
)

# Step 7: Make predictions
print("\nMaking predictions...")
test_texts = [
    "This is the best thing I've ever bought!",
    "Completely useless, don't buy this.",
    "Pretty decent for the price.",
]

predictions = classifier.predict(test_texts)
probabilities = classifier.predict_proba(test_texts)

# Display results
print("\nPredictions:")
for text, pred, proba in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = proba[pred]
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
```

## Understanding the Code

Let's break down each step:

### Step 1: Prepare Training Data

```python
texts = ["I love this product!", "Terrible experience", ...]
labels = [1, 0, ...]  # Binary labels
```

- `texts`: List of text samples
- `labels`: Corresponding labels (0 or 1 for binary classification)

### Step 2: Train Tokenizer

```python
tokenizer = WordPieceTokenizer()
tokenizer.train(texts, vocab_size=500, min_frequency=1)
```

The tokenizer learns to split text into subwords:
- `vocab_size`: Maximum vocabulary size
- `min_frequency`: Minimum frequency for a token to be included

### Step 3: Configure Model

```python
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=2,
)
```

- `embedding_dim`: Dimension of the embedding vectors
- `num_classes`: Number of output classes (2 for binary classification)

### Step 4: Configure Training

```python
training_config = TrainingConfig(
    num_epochs=10,
    batch_size=4,
    lr=1e-3,
    patience_early_stopping=5,
)
```

- `num_epochs`: Maximum number of training epochs
- `batch_size`: Number of samples per batch
- `lr`: Learning rate
- `patience_early_stopping`: Stop if validation loss doesn't improve for this many epochs

### Step 5-6: Create and Train

```python
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

classifier.train(X_text=texts, y=labels, training_config=training_config)
```

The classifier orchestrates the entire training process using PyTorch Lightning.

### Step 7: Make Predictions

```python
predictions = classifier.predict(test_texts)
probabilities = classifier.predict_proba(test_texts)
```

- `predict()`: Returns class predictions
- `predict_proba()`: Returns class probabilities

## Expected Output

When you run this example, you should see output similar to:

```
Training tokenizer...
Tokenizer trained with vocabulary size: 245

Creating classifier...

Training model...
Epoch 0: 100%|██████████| 3/3 [00:00<00:00, 15.23it/s, v_num=0]
Epoch 1: 100%|██████████| 3/3 [00:00<00:00, 18.45it/s, v_num=0]
...

Making predictions...

Predictions:

Text: This is the best thing I've ever bought!
Sentiment: Positive (confidence: 92.34%)

Text: Completely useless, don't buy this.
Sentiment: Negative (confidence: 88.76%)

Text: Pretty decent for the price.
Sentiment: Positive (confidence: 65.43%)
```

## Running with Your Own Data

To use your own data, simply replace the `texts` and `labels` with your dataset:

```python
# Your own data
texts = [...]  # List of strings
labels = [...]  # List of integers (0, 1, 2, ... for multiclass)

# For multiclass classification (e.g., 3 classes)
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=3,  # Change this to your number of classes
)
```

## Using Validation Data

For better model evaluation, split your data into training and validation sets:

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Train with validation
classifier.train(
    X_text=X_train,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    training_config=training_config,
)
```

## What's Next?

Now that you've built your first classifier, you can:

- **Explore tutorials**: See {doc}`../tutorials/index` for more advanced examples
- **Understand the architecture**: Read {doc}`../architecture/overview` to learn how it works
- **Customize your model**: Check the {doc}`../api/index` for all configuration options
- **Add categorical features**: See {doc}`../tutorials/index` for combining text with other data

## Common Issues

### Small Dataset Warning

If you see warnings about small datasets, that's expected for this quick example. For real applications, use larger datasets (hundreds or thousands of samples).

### Training on GPU

To use GPU acceleration:

```python
training_config = TrainingConfig(
    ...
    accelerator="gpu",  # or "mps" for Mac M1/M2
)
```

### Reproducibility

For reproducible results, set seeds:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## Summary

In this quick start, you:

- ✅ Trained a WordPiece tokenizer
- ✅ Configured a text classification model
- ✅ Trained the model with PyTorch Lightning
- ✅ Made predictions on new text

You're now ready to explore more advanced features and build production-ready classifiers!
