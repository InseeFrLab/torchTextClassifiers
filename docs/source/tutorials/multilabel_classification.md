# Multilabel Classification

Learn how to assign multiple labels to each text sample, enabling more complex classification scenarios.

## Learning Objectives

By the end of this tutorial, you'll be able to:

- Understand multilabel vs. multiclass classification
- Use both ragged-list and one-hot encoding approaches
- Configure appropriate loss functions for multilabel tasks
- Evaluate multilabel predictions
- Handle variable numbers of labels per sample

## Prerequisites

- Completed {doc}`multiclass_classification` tutorial
- Understanding of binary classification
- Familiarity with numpy arrays

## Multilabel vs. Multiclass

### Multiclass Classification

Each sample has **exactly one label** from multiple classes:

```python
texts = ["Sports article", "Tech news", "Business report"]
labels = [0, 1, 2]  # Each sample has ONE label
```

### Multilabel Classification

Each sample can have **zero, one, or multiple labels**:

```python
texts = [
    "Article about AI in healthcare",  # Both Tech AND Health
    "Sports news from Europe",          # Both Sports AND Europe
    "Local business report"             # Just Business
]

# Multiple labels per sample
labels = [
    [1, 3],      # Tech (1) + Health (3)
    [0, 4],      # Sports (0) + Europe (4)
    [2]          # Business (2) only
]
```

### Real-World Use Cases

✅ **Document tagging**: Article can have multiple topics
✅ **Product categorization**: Product can belong to multiple categories
✅ **Symptom detection**: Patient can have multiple symptoms
✅ **Content moderation**: Content can violate multiple rules
✅ **Multi-genre classification**: Movie can have multiple genres

## Two Approaches to Multilabel

### Approach 1: Ragged Lists (Recommended)

Each sample has a **list of label indices**:

```python
labels = [
    [0, 1, 5],   # Sample has labels 0, 1, and 5
    [0, 4],      # Sample has labels 0 and 4
    [1, 5],      # Sample has labels 1 and 5
]
```

**Pros:**
- Natural representation
- Saves memory
- Easy to construct

**Cons:**
- Can't directly convert to numpy array
- Variable-length lists

### Approach 2: One-Hot Encoding

Each sample has a **binary vector** (1 = label present, 0 = absent):

```python
labels = [
    [1, 1, 0, 0, 0, 1],  # Labels 0, 1, 5 present
    [1, 0, 0, 0, 1, 0],  # Labels 0, 4 present
    [0, 1, 0, 0, 0, 1],  # Labels 1, 5 present
]
```

**Pros:**
- Fixed-size numpy array
- Can store probabilities (not just 0/1)

**Cons:**
- Memory-intensive for many labels
- Sparse representation

## Complete Example: Ragged Lists

```python
import numpy as np
import torch
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Sample data: Each text can have multiple labels
texts = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case",
    "Good example here",
    "Bad example here"
]

# Ragged lists: Variable-length label lists
labels = [
    [0, 1, 5],   # Has 3 labels
    [0, 4],      # Has 2 labels
    [1, 5],      # Has 2 labels
    [0, 1, 4],   # Has 3 labels
    [1, 5],      # Has 2 labels
    [0]          # Has 1 label
]

# Prepare data
X = np.array(texts)
y = np.array(labels, dtype=object)  # dtype=object for ragged lists

# Create tokenizer
tokenizer = WordPieceTokenizer(vocab_size=1000)
tokenizer.train(X.tolist())

# Calculate number of classes
num_classes = max(max(label_list) for label_list in labels) + 1

# Configure model
model_config = ModelConfig(
    embedding_dim=96,
    num_classes=num_classes
)

# IMPORTANT: Use BCEWithLogitsLoss for multilabel
training_config = TrainingConfig(
    lr=1e-3,
    batch_size=4,
    num_epochs=10,
    loss=torch.nn.BCEWithLogitsLoss()  # Multilabel loss
)

# Create classifier with ragged_multilabel=True
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=True  # Key parameter!
)

# Train
classifier.train(
    X_train=X,
    y_train=y,
    training_config=training_config
)

# Predict
result = classifier.predict(X)
predictions = result["prediction"]
```

## Complete Example: One-Hot Encoding

```python
import numpy as np
import torch
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Same texts
texts = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case",
    "Good example here",
    "Bad example here"
]

# One-hot encoding: Binary vectors
# 6 samples, 6 possible labels (0-5)
labels = [
    [1., 1., 0., 0., 0., 1.],  # Labels 0, 1, 5 present
    [1., 0., 0., 0., 1., 0.],  # Labels 0, 4 present
    [0., 1., 0., 0., 0., 1.],  # Labels 1, 5 present
    [1., 1., 0., 0., 1., 0.],  # Labels 0, 1, 4 present
    [0., 1., 0., 0., 0., 1.],  # Labels 1, 5 present
    [1., 0., 0., 0., 0., 0.]   # Label 0 present
]

# Prepare data
X = np.array(texts)
y = np.array(labels)  # Can convert to numpy array now!

# Create tokenizer
tokenizer = WordPieceTokenizer(vocab_size=1000)
tokenizer.train(X.tolist())

# Configure model
num_classes = y.shape[1]  # Number of columns

model_config = ModelConfig(
    embedding_dim=96,
    num_classes=num_classes
)

training_config = TrainingConfig(
    lr=1e-3,
    batch_size=4,
    num_epochs=10,
    loss=torch.nn.BCEWithLogitsLoss()
)

# Create classifier with ragged_multilabel=False (default)
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=False  # or omit (default is False)
)

# Train
classifier.train(
    X_train=X,
    y_train=y,
    training_config=training_config
)

# Predict
result = classifier.predict(X)
predictions = result["prediction"]
```

## Step-by-Step Walkthrough

### 1. Choose Your Approach

**Use Ragged Lists if:**
- You have variable numbers of labels per sample
- Memory is a concern
- Data is naturally in list format

**Use One-Hot if:**
- You need fixed-size arrays
- You want to store probabilities
- You're integrating with systems expecting one-hot

### 2. Prepare Labels

#### Ragged Lists

```python
# List of lists (variable length)
labels = [[0, 1], [1, 2, 3], [0]]

# Convert to numpy array with dtype=object
y = np.array(labels, dtype=object)
```

#### One-Hot Encoding

```python
# Manual creation
labels = [
    [1, 0, 0, 0],  # Label 0
    [0, 1, 1, 1],  # Labels 1, 2, 3
    [1, 0, 0, 0]   # Label 0
]

# Or convert from ragged lists
from sklearn.preprocessing import MultiLabelBinarizer

ragged_labels = [[0, 1], [1, 2, 3], [0]]
mlb = MultiLabelBinarizer()
one_hot_labels = mlb.fit_transform(ragged_labels)
```

### 3. Configure Loss Function

**Always use `BCEWithLogitsLoss` for multilabel:**

```python
import torch

training_config = TrainingConfig(
    # ... other params ...
    loss=torch.nn.BCEWithLogitsLoss()
)
```

**Why not CrossEntropyLoss?**
- `CrossEntropyLoss`: Classes compete (only one can win)
- `BCEWithLogitsLoss`: Each label is independent binary decision

### 4. Set ragged_multilabel Flag

```python
# For ragged lists
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=True  # Must be True
)

# For one-hot encoding
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=False  # Must be False (default)
)
```

**Warning:** Setting wrong flag leads to incorrect behavior!

### 5. Understanding Predictions

#### Ragged Lists

Predictions are **probability scores** for each label:

```python
result = classifier.predict(X_test)
predictions = result["prediction"]  # Shape: (n_samples, n_classes)

# For each sample, check which labels are active
threshold = 0.5
for i, pred in enumerate(predictions):
    active_labels = [j for j, prob in enumerate(pred) if prob > threshold]
    print(f"Sample {i}: {active_labels}")
```

#### One-Hot Encoding

Same format - probabilities for each label:

```python
predictions = result["prediction"]  # Shape: (n_samples, n_classes)

# Apply threshold
predicted_labels = (predictions > 0.5).astype(int)
```

## Evaluation Metrics

### Exact Match Accuracy

All labels must match exactly:

```python
def exact_match_accuracy(y_true, y_pred, threshold=0.5):
    """Calculate exact match accuracy."""
    y_pred_binary = (y_pred > threshold).astype(int)

    # Check if each sample matches exactly
    matches = np.all(y_pred_binary == y_true, axis=1)
    return matches.mean()

accuracy = exact_match_accuracy(y_test, predictions)
print(f"Exact Match Accuracy: {accuracy:.3f}")
```

### Hamming Loss

Average per-label error:

```python
from sklearn.metrics import hamming_loss

# Convert predictions to binary
y_pred_binary = (predictions > 0.5).astype(int)

loss = hamming_loss(y_test, y_pred_binary)
print(f"Hamming Loss: {loss:.3f}")  # Lower is better
```

### F1 Score

Harmonic mean of precision and recall:

```python
from sklearn.metrics import f1_score

# Micro: Calculate globally
f1_micro = f1_score(y_test, y_pred_binary, average='micro')

# Macro: Average per label
f1_macro = f1_score(y_test, y_pred_binary, average='macro')

# Weighted: Weighted by support
f1_weighted = f1_score(y_test, y_pred_binary, average='weighted')

print(f"F1 Micro: {f1_micro:.3f}")
print(f"F1 Macro: {f1_macro:.3f}")
print(f"F1 Weighted: {f1_weighted:.3f}")
```

### Subset Accuracy

Same as exact match accuracy:

```python
from sklearn.metrics import accuracy_score

subset_acc = accuracy_score(y_test, y_pred_binary)
print(f"Subset Accuracy: {subset_acc:.3f}")
```

## Real-World Example: Document Tagging

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Document tagging dataset
texts = [
    "Python tutorial for machine learning",
    "Introduction to neural networks",
    "Web development with JavaScript",
    "Data visualization with Python",
    "Deep learning research paper",
    "Building REST APIs in Python"
]

# Labels: 0=Programming, 1=AI/ML, 2=Web, 3=Data, 4=Research
labels = [
    [0, 1],      # Programming + AI/ML
    [1, 4],      # AI/ML + Research
    [0, 2],      # Programming + Web
    [0, 3],      # Programming + Data
    [1, 4],      # AI/ML + Research
    [0, 2]       # Programming + Web
]

# Prepare data
X = np.array(texts)
y = np.array(labels, dtype=object)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train model
tokenizer = WordPieceTokenizer(vocab_size=1000)
tokenizer.train(X_train.tolist())

num_classes = 5

model_config = ModelConfig(
    embedding_dim=64,
    num_classes=num_classes
)

training_config = TrainingConfig(
    lr=1e-3,
    batch_size=2,
    num_epochs=50,
    loss=torch.nn.BCEWithLogitsLoss()
)

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    ragged_multilabel=True
)

classifier.train(X_train, y_train, training_config=training_config)

# Predict and evaluate
result = classifier.predict(X_test)
predictions = result["prediction"]

# Convert to binary predictions
y_pred_binary = (predictions > 0.5).astype(int)

# Convert ragged y_test to one-hot for evaluation
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=range(num_classes))
y_test_binary = mlb.fit_transform(y_test)

# Evaluate
from sklearn.metrics import classification_report

label_names = ['Programming', 'AI/ML', 'Web', 'Data', 'Research']
print(classification_report(
    y_test_binary,
    y_pred_binary,
    target_names=label_names
))
```

## Common Issues

### Issue 1: Wrong ragged_multilabel Setting

**Error:** Model trains but predictions are incorrect

**Solution:** Ensure flag matches your data format:
```python
# Ragged lists → ragged_multilabel=True
# One-hot → ragged_multilabel=False
```

### Issue 2: Using CrossEntropyLoss

**Problem:** Model doesn't learn properly

**Solution:** Always use `BCEWithLogitsLoss`:
```python
training_config = TrainingConfig(
    loss=torch.nn.BCEWithLogitsLoss()
)
```

### Issue 3: Shape Mismatch

**Error:** "Expected 2D array for labels"

**Solution:** For ragged lists, use `dtype=object`:
```python
y = np.array(labels, dtype=object)
```

### Issue 4: All Predictions Same

**Possible causes:**
- Not enough training data
- Learning rate too high/low
- Class imbalance

**Try:**
- Increase training epochs
- Adjust learning rate
- Check label distribution

## Customization

### Custom Threshold

Adjust sensitivity vs. precision:

```python
# Conservative (higher precision)
threshold = 0.7
predicted_labels = (predictions > threshold).astype(int)

# Aggressive (higher recall)
threshold = 0.3
predicted_labels = (predictions > threshold).astype(int)
```

### Class Weights

Handle imbalanced labels:

```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

# For one-hot labels
class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(num_classes),
    y=y_train.argmax(axis=1)  # Works for imbalanced data
)

# Use in loss (requires custom loss function)
```

### With Attention

For long documents:

```python
from torchTextClassifiers.model.components import AttentionConfig

attention_config = AttentionConfig(
    n_embd=128,
    n_head=8,
    n_layer=3
)

model_config = ModelConfig(
    embedding_dim=128,
    num_classes=num_classes,
    attention_config=attention_config
)
```

## Advanced: Probabilistic Labels

One-hot encoding supports probabilities:

```python
# Soft labels (not just 0 or 1)
labels = [
    [0.9, 0.8, 0.1, 0.0, 0.0, 0.7],  # Confident in 0,1,5
    [0.6, 0.0, 0.0, 0.0, 0.5, 0.0],  # Less confident in 0,4
]

y = np.array(labels)  # Probabilities between 0 and 1

# Use same setup, BCEWithLogitsLoss handles probabilities
```

## Best Practices

1. **Choose the right approach:** Ragged lists for most cases, one-hot for probabilities
2. **Always use BCEWithLogitsLoss:** Essential for multilabel
3. **Set ragged_multilabel correctly:** Matches your data format
4. **Use appropriate metrics:** F1, Hamming loss better than accuracy
5. **Tune threshold:** Balance precision vs. recall for your use case
6. **Handle imbalance:** Common in multilabel - consider class weights

## Summary

**Key takeaways:**
- Multilabel: Each sample can have multiple labels
- Two approaches: Ragged lists (recommended) or one-hot encoding
- Always use `BCEWithLogitsLoss` for multilabel tasks
- Set `ragged_multilabel=True` for ragged lists
- Evaluate with F1, Hamming loss, or exact match accuracy

Ready to combine everything? Try adding categorical features to multilabel classification, or use explainability to understand multilabel predictions!

## Next Steps

- **Mixed features**: Combine multilabel with categorical features
- **Explainability**: Understand which words trigger which labels
- **API Reference**: See {doc}`../api/index` for detailed documentation
