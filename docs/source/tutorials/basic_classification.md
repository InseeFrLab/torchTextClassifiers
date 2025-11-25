# Binary Classification Tutorial

Learn how to build a binary sentiment classifier for product reviews.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Create and train a WordPiece tokenizer
- Configure a binary classification model
- Train the model with validation data
- Make predictions and evaluate performance
- Understand the complete workflow from data to predictions

## Prerequisites

- Basic Python knowledge
- torchTextClassifiers installed
- Familiarity with classification concepts

## Overview

In this tutorial, we'll build a **sentiment classifier** that predicts whether a product review is positive or negative. We'll use:

- **Dataset**: Product reviews (30 training, 8 validation, 10 test samples)
- **Task**: Binary classification (positive vs. negative)
- **Tokenizer**: WordPiece
- **Architecture**: Simple text embedder + classification head

## Complete Code

Here's the complete code we'll walk through:

```python
import os
import numpy as np
import torch
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# For Mac M1/M2 users
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Step 1: Prepare Data
X_train = np.array([
    "I love this product! It's amazing and works perfectly.",
    "This is terrible. Worst purchase ever made.",
    "Great quality and fast shipping. Highly recommend!",
    "Poor quality, broke after one day. Very disappointed.",
    "Excellent customer service and great value for money.",
    "Overpriced and doesn't work as advertised.",
    # ... (30 total samples)
])
y_train = np.array([1, 0, 1, 0, 1, 0, ...])  # 1=positive, 0=negative

X_val = np.array([
    "Good product, satisfied with purchase.",
    "Not worth the money, poor quality.",
    # ... (8 total samples)
])
y_val = np.array([1, 0, ...])

X_test = np.array([
    "This is an amazing product with great features!",
    "Completely disappointed with this purchase.",
    # ... (10 total samples)
])
y_test = np.array([1, 0, ...])

# Step 2: Create and Train Tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
tokenizer.train(X_train.tolist())

# Step 3: Configure Model
model_config = ModelConfig(
    embedding_dim=50,
    num_classes=2
)

# Step 4: Create Classifier
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# Step 5: Train Model
training_config = TrainingConfig(
    num_epochs=20,
    batch_size=4,
    lr=1e-3,
    patience_early_stopping=5,
    num_workers=0,
)

classifier.train(
    X_train, y_train,
    X_val, y_val,
    training_config=training_config,
    verbose=True
)

# Step 6: Make Predictions
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()
confidence = result["confidence"].squeeze().numpy()

# Step 7: Evaluate
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.3f}")
```

## Step-by-Step Walkthrough

### Step 1: Prepare Your Data

First, organize your data into training, validation, and test sets:

```python
X_train = np.array([
    "I love this product! It's amazing and works perfectly.",
    "This is terrible. Worst purchase ever made.",
    # ... more samples
])
y_train = np.array([1, 0, ...])  # Binary labels
```

**Key Points:**

- **Training set**: Used to train the model (30 samples)
- **Validation set**: Used for early stopping and hyperparameter tuning (8 samples)
- **Test set**: Used for final evaluation (10 samples)
- **Labels**: 0 = negative, 1 = positive

:::{tip}
For real projects, use at least hundreds of samples per class. This example uses small numbers for demonstration.
:::

### Step 2: Create and Train Tokenizer

The tokenizer converts text into numerical tokens:

```python
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
tokenizer.train(X_train.tolist())
```

**Parameters:**

- `vocab_size`: Maximum vocabulary size (5000 subwords)
- `output_dim`: Output dimension for tokenized sequences (128 tokens max)

**What happens during training:**

1. Analyzes the training corpus
2. Learns common subwords and character combinations
3. Builds a vocabulary of frequent patterns

:::{note}
The tokenizer only sees the training data, never validation or test data, to avoid data leakage.
:::

### Step 3: Configure the Model

Define your model architecture:

```python
model_config = ModelConfig(
    embedding_dim=50,
    num_classes=2
)
```

**Parameters:**

- `embedding_dim`: Dimension of learned text embeddings (50)
- `num_classes`: Number of output classes (2 for binary classification)

**Architecture:**

The model will have:
- Embedding layer: Maps tokens to 50-dimensional vectors
- Pooling: Averages token embeddings
- Classification head: Linear layer outputting 2 logits

### Step 4: Create the Classifier

Instantiate the classifier with the tokenizer and configuration:

```python
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)
```

This creates the complete pipeline: tokenizer → embedder → classifier.

### Step 5: Configure and Run Training

Set up training hyperparameters:

```python
training_config = TrainingConfig(
    num_epochs=20,              # Maximum training epochs
    batch_size=4,               # Samples per batch
    lr=1e-3,                    # Learning rate
    patience_early_stopping=5,  # Stop if no improvement for 5 epochs
    num_workers=0,              # Data loading workers
)
```

**Key Hyperparameters:**

- **num_epochs**: How many times to iterate through the dataset
- **batch_size**: Smaller = more updates but slower; larger = faster but less stable
- **lr (learning rate)**: How big the optimization steps are
- **patience_early_stopping**: Prevents overfitting by stopping early

Train the model:

```python
classifier.train(
    X_train, y_train,    # Training data
    X_val, y_val,        # Validation data
    training_config=training_config,
    verbose=True         # Show training progress
)
```

**Expected Output:**

```
Epoch 0: 100%|██████████| 8/8 [00:00<00:00, 25.32it/s, v_num=0]
Epoch 1: 100%|██████████| 8/8 [00:00<00:00, 28.41it/s, v_num=0]
...
```

:::{tip}
Watch the validation metrics during training. If validation loss increases while training loss decreases, you may be overfitting.
:::

### Step 6: Make Predictions

Use the trained model to predict on new data:

```python
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()
confidence = result["confidence"].squeeze().numpy()
```

**Output:**

- `predictions`: Predicted class labels (0 or 1)
- `confidence`: Confidence scores (0-1 range)

**Example output:**

```python
predictions = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
confidence = [0.95, 0.88, 0.92, 0.76, 0.98, 0.85, 0.91, 0.79, 0.94, 0.87]
```

### Step 7: Evaluate Performance

Calculate accuracy:

```python
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.3f}")
```

Show detailed results:

```python
for i, (text, pred, true) in enumerate(zip(X_test, predictions, y_test)):
    sentiment = "Positive" if pred == 1 else "Negative"
    correct = "✅" if pred == true else "❌"
    print(f"{i+1}. {correct} Predicted: {sentiment}")
    print(f"   Text: {text[:50]}...")
```

**Example output:**

```
1. ✅ Predicted: Positive
   Text: This is an amazing product with great features...

2. ✅ Predicted: Negative
   Text: Completely disappointed with this purchase...

Test accuracy: 0.900
```

## Understanding the Results

### What Does Good Performance Look Like?

- **Accuracy > 0.80**: Good for simple binary classification
- **Accuracy > 0.90**: Excellent performance
- **Confidence scores high**: Model is certain about predictions

### When to Worry

- **Accuracy < 0.60**: Model barely better than random guessing
- **Validation loss increasing**: Possible overfitting
- **Low confidence scores**: Model is uncertain

## Customization Options

### Using Different Tokenizers

Try the NGramTokenizer (FastText-style):

```python
from torchTextClassifiers.tokenizers import NGramTokenizer

tokenizer = NGramTokenizer(
    vocab_size=5000,
    min_n=3,  # Minimum n-gram size
    max_n=6,  # Maximum n-gram size
)
tokenizer.train(X_train.tolist())
```

### Adjusting Model Size

For better performance with more data:

```python
model_config = ModelConfig(
    embedding_dim=128,  # Larger embeddings
    num_classes=2
)
```

### Training Longer

```python
training_config = TrainingConfig(
    num_epochs=50,              # More epochs
    batch_size=16,              # Larger batches
    lr=5e-4,                    # Lower learning rate
    patience_early_stopping=10, # More patience
)
```

### Using GPU

If you have a GPU:

```python
training_config = TrainingConfig(
    ...
    accelerator="gpu",  # Use GPU
)
```

## Common Issues and Solutions

### Issue: Low Accuracy

**Solutions:**

1. Increase `embedding_dim` (e.g., 128 or 256)
2. Train for more epochs
3. Collect more training data
4. Try different learning rates (1e-4, 5e-4, 1e-3)

### Issue: Model Overfitting

**Symptoms:** High training accuracy, low validation accuracy

**Solutions:**

1. Reduce `embedding_dim`
2. Add more training data
3. Reduce `patience_early_stopping` for earlier stopping
4. Use data augmentation

### Issue: Training Too Slow

**Solutions:**

1. Increase `batch_size` (if memory allows)
2. Reduce `num_epochs`
3. Use `accelerator="gpu"`
4. Increase `num_workers` (for data loading)

## Next Steps

Now that you've built a binary classifier, you can:

1. **Try multiclass classification**: See {doc}`multiclass_classification`
2. **Add categorical features**: Learn about mixed features
3. **Use explainability**: Understand which words drive predictions
4. **Explore architecture**: Read {doc}`../architecture/overview`

## Complete Working Example

You can find the complete working example in the repository:
- [examples/basic_classification.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/basic_classification.py)

## Summary

In this tutorial, you learned:

- ✅ How to prepare training, validation, and test data
- ✅ How to create and train a WordPiece tokenizer
- ✅ How to configure a binary classification model
- ✅ How to train the model with early stopping
- ✅ How to make predictions and evaluate performance
- ✅ How to customize hyperparameters

You're now ready to build your own text classifiers!
