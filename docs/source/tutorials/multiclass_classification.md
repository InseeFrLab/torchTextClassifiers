# Multiclass Classification Tutorial

Learn how to build a multiclass sentiment classifier with 3 classes: negative, neutral, and positive.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Handle multiclass classification problems (3+ classes)
- Configure models for multiple output classes
- Ensure reproducible results with proper seeding
- Evaluate multiclass performance
- Understand class distribution and balance

## Prerequisites

- Completion of {doc}`basic_classification` tutorial (recommended)
- Basic understanding of classification
- torchTextClassifiers installed

## Overview

In this tutorial, we'll build a **3-class sentiment classifier** that categorizes product reviews as:

- **Negative** (class 0): Bad reviews
- **Neutral** (class 1): Mixed or moderate reviews
- **Positive** (class 2): Good reviews

**Key Difference from Binary Classification:**

- Binary: 2 classes (positive/negative)
- Multiclass: 3+ classes (negative/neutral/positive)

## Complete Code

```python
import os
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Step 1: Set Seeds for Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
seed_everything(SEED, workers=True)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Step 2: Prepare Multi-class Data
X_train = np.array([
    # Negative (class 0)
    "This product is terrible and I hate it completely.",
    "Worst purchase ever. Total waste of money.",
    "Absolutely awful quality. Very disappointed.",
    "Poor service and terrible product quality.",
    "I regret buying this. Complete failure.",

    # Neutral (class 1)
    "The product is okay, nothing special though.",
    "It works but could be better designed.",
    "Average quality for the price point.",
    "Not bad but not great either.",
    "It's fine, meets basic expectations.",

    # Positive (class 2)
    "Excellent product! Highly recommended!",
    "Amazing quality and great customer service.",
    "Perfect! Exactly what I was looking for.",
    "Outstanding value and excellent performance.",
    "Love it! Will definitely buy again."
])

y_train = np.array([0, 0, 0, 0, 0,  # negative
                    1, 1, 1, 1, 1,  # neutral
                    2, 2, 2, 2, 2]) # positive

# Validation data
X_val = np.array([
    "Bad quality, not recommended.",
    "It's okay, does the job.",
    "Great product, very satisfied!"
])
y_val = np.array([0, 1, 2])

# Test data
X_test = np.array([
    "This is absolutely horrible!",
    "It's an average product, nothing more.",
    "Fantastic! Love every aspect of it!",
    "Really poor design and quality.",
    "Works well, good value for money.",
    "Outstanding product with amazing features!"
])
y_test = np.array([0, 1, 2, 0, 1, 2])

# Step 3: Create and Train Tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
tokenizer.train(X_train.tolist())

# Step 4: Configure Model for 3 Classes
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=3  # KEY: 3 classes for multiclass
)

# Step 5: Create Classifier
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# Step 6: Train Model
training_config = TrainingConfig(
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    patience_early_stopping=7,
    num_workers=0,
    trainer_params={'deterministic': True}
)

classifier.train(
    X_train, y_train,
    X_val, y_val,
    training_config=training_config,
    verbose=True
)

# Step 7: Make Predictions
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()

# Step 8: Evaluate
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.3f}")

# Show results with class names
class_names = ["Negative", "Neutral", "Positive"]
for text, pred, true in zip(X_test, predictions, y_test):
    predicted = class_names[pred]
    actual = class_names[true]
    status = "✅" if pred == true else "❌"
    print(f"{status} Predicted: {predicted}, True: {actual}")
    print(f"   Text: {text}")
```

## Step-by-Step Walkthrough

### Step 1: Ensuring Reproducibility

For consistent results across runs, set seeds properly:

```python
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
seed_everything(SEED, workers=True)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

**Why this matters:**

- Makes experiments reproducible
- Enables fair comparison of hyperparameters
- Helps debug model behavior

:::{tip}
Always set seeds when reporting results or comparing models!
:::

### Step 2: Preparing Multiclass Data

Unlike binary classification, you now have **3 classes**:

```python
y_train = np.array([0, 0, 0, 0, 0,  # negative (class 0)
                    1, 1, 1, 1, 1,  # neutral (class 1)
                    2, 2, 2, 2, 2]) # positive (class 2)
```

**Important:** Class labels should be:
- **Integers**: 0, 1, 2, ... (not strings)
- **Continuous**: Start from 0, no gaps (0, 1, 2 not 0, 2, 5)
- **Balanced**: Ideally equal samples per class

**Check class distribution:**

```python
print(f"Negative: {sum(y_train==0)}")
print(f"Neutral: {sum(y_train==1)}")
print(f"Positive: {sum(y_train==2)}")
```

Output:
```
Negative: 5
Neutral: 5
Positive: 5
```

:::{note}
This example has perfectly balanced classes (5 samples each). Real datasets are often imbalanced.
:::

### Step 3-4: Model Configuration

The **only** difference from binary classification:

```python
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=3  # Change from 2 to 3
)
```

**Under the hood:**

- Binary: Uses 2 output neurons + CrossEntropyLoss
- Multiclass: Uses 3 output neurons + CrossEntropyLoss
- The loss function handles both automatically!

### Step 5-6: Training

Training is identical to binary classification:

```python
classifier.train(
    X_train, y_train,
    X_val, y_val,
    training_config=training_config
)
```

**Training process:**

1. Forward pass: Text → Embeddings → Logits (3 values)
2. Loss calculation: Compare logits to true labels
3. Backward pass: Compute gradients
4. Update weights: Optimizer step
5. Repeat for each batch/epoch

### Step 7: Making Predictions

Predictions now return values in {0, 1, 2}:

```python
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()
# Example: [0, 1, 2, 0, 1, 2]
```

**Probability interpretation:**

You can also get probabilities for each class:

```python
probabilities = result["confidence"].squeeze().numpy()
# Shape: (num_samples, 3)
# Each row sums to 1.0
```

### Step 8: Evaluation

For multiclass, use class names for clarity:

```python
class_names = ["Negative", "Neutral", "Positive"]

for pred, true in zip(predictions, y_test):
    predicted_label = class_names[pred]
    true_label = class_names[true]
    print(f"Predicted: {predicted_label}, True: {true_label}")
```

**Output:**

```
✅ Predicted: Negative, True: Negative
   Text: This is absolutely horrible!

✅ Predicted: Neutral, True: Neutral
   Text: It's an average product, nothing more.

✅ Predicted: Positive, True: Positive
   Text: Fantastic! Love every aspect of it!
```

## Advanced: Class Imbalance

Real datasets often have unbalanced classes:

```python
# Imbalanced example
y_train = [0]*100 + [1]*20 + [2]*10  # 100:20:10 ratio
```

**Solutions:**

### 1. Class Weights

Weight the loss function to penalize minority class errors more:

```python
from torch import nn

# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize

# Use weighted loss
training_config = TrainingConfig(
    ...
    loss=nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
)
```

### 2. Oversampling/Undersampling

Balance the dataset before training:

```python
from sklearn.utils import resample

# Oversample minority classes or undersample majority class
# (Use before creating the classifier)
```

### 3. Data Augmentation

Generate synthetic samples for minority classes.

## Evaluation Metrics

For multiclass problems, accuracy isn't enough. Use:

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
#       Pred 0  Pred 1  Pred 2
# True 0 [[  2       0       0]
# True 1  [  0       2       0]
# True 2  [  0       0       2]]
```

### Classification Report

```python
report = classification_report(
    y_test, predictions,
    target_names=["Negative", "Neutral", "Positive"]
)
print(report)
```

**Output:**

```
              precision    recall  f1-score   support

    Negative       1.00      1.00      1.00         2
     Neutral       1.00      1.00      1.00         2
    Positive       1.00      1.00      1.00         2

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6
```

**Metrics explained:**

- **Precision**: Of predicted class X, how many were correct?
- **Recall**: Of true class X, how many did we find?
- **F1-score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

## Extending to More Classes

For 5 classes (e.g., star ratings 1-5):

```python
# Data with 5 classes
y_train = np.array([0, 1, 2, 3, 4, ...])  # 0=1-star, 4=5-star

# Model configuration
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=5  # Change to 5
)
```

The same code works for any number of classes!

## Common Issues

### Issue: Poor Performance on Middle Classes

**Problem:** Neutral class has low accuracy

**Solution:**

1. Collect more neutral examples
2. Make the distinction clearer in your data
3. Consider if neutral is necessary (binary might be better)

### Issue: Model Always Predicts One Class

**Symptoms:** All predictions are class 0 or class 2

**Solutions:**

1. Check class balance - might be too imbalanced
2. Verify labels are correct (0, 1, 2 not 1, 2, 3)
3. Lower learning rate
4. Train for more epochs

### Issue: Overfitting

**Symptoms:** High training accuracy, low test accuracy

**Solutions:**

1. Reduce `embedding_dim`
2. Add more training data
3. Use stronger early stopping (lower `patience`)

## Next Steps

Now that you understand multiclass classification:

1. **Add categorical features**: Combine text with metadata
2. **Try multilabel classification**: Multiple labels per sample
3. **Use explainability**: See which words matter for each class
4. **Explore advanced architectures**: Add attention mechanisms

## Complete Working Example

Find the full code in the repository:
- [examples/multiclass_classification.py](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/examples/multiclass_classification.py)

## Summary

In this tutorial, you learned:

- ✅ How to set up multiclass classification (3+ classes)
- ✅ How to configure `num_classes` correctly
- ✅ How to ensure reproducible results with proper seeding
- ✅ How to check and handle class distribution
- ✅ How to evaluate multiclass models with confusion matrices
- ✅ How to handle class imbalance

You're now ready to tackle real-world multiclass problems!
