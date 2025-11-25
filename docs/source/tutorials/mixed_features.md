# Mixed Features Classification

Learn how to combine text with categorical variables for improved classification performance.

## Learning Objectives

By the end of this tutorial, you'll be able to:

- Combine text and categorical features in a single model
- Configure categorical embeddings
- Compare performance with and without categorical features
- Understand when categorical features improve results

## Prerequisites

- Completed {doc}`basic_classification` tutorial
- Familiarity with categorical data (e.g., user demographics, product categories)
- Understanding of embeddings

## What Are Categorical Features?

Categorical features are non-numeric variables like:
- **User attributes**: Age group, location, membership tier
- **Product metadata**: Category, brand, seller
- **Document properties**: Source, type, language

These features can significantly improve classification when they contain relevant information.

## When to Use Categorical Features

✅ **Good use cases:**
- Product descriptions + (category, brand)
- Reviews + (user location, verified purchase)
- News articles + (source, publication date)

❌ **Poor use cases:**
- Text already contains the categorical information
- Random or high-cardinality features (e.g., user IDs)
- Categorical features with no relationship to labels

## Complete Example

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Sample data: Product reviews with category
texts = [
    "Great phone with excellent camera",
    "Battery dies too quickly",
    "Love this laptop's performance",
    "Screen quality is poor",
    "Best headphones I've ever owned",
    "Sound quality is disappointing",
    "Fast shipping and great quality",
    "Product arrived damaged"
]

# Categorical feature: Product category (0=Electronics, 1=Audio)
categories = [0, 0, 0, 0, 1, 1, 0, 0]

# Labels: Positive (1) or Negative (0)
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# Prepare data
X_text = np.array(texts)
X_categorical = np.array(categories).reshape(-1, 1)  # Shape: (n_samples, 1)
y = np.array(labels)

# Split data
X_text_train, X_text_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_text, X_categorical, y, test_size=0.25, random_state=42
)

# Create tokenizer
tokenizer = WordPieceTokenizer(vocab_size=1000)
tokenizer.train(X_text_train.tolist())

# Configure model WITH categorical features
model_config = ModelConfig(
    embedding_dim=64,
    num_classes=2,
    categorical_vocabulary_sizes=[2],  # 2 categories (Electronics, Audio)
    categorical_embedding_dims=[8],     # Embed each category into 8 dimensions
)

# Create classifier
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# Training configuration
training_config = TrainingConfig(
    num_epochs=20,
    batch_size=4,
    lr=1e-3
)

# Combine text and categorical features
X_train_mixed = np.column_stack([X_text_train, X_cat_train])
X_test_mixed = np.column_stack([X_text_test, X_cat_test])

# Train model
classifier.train(
    X_train_mixed, y_train,
    training_config=training_config
)

# Predict
result = classifier.predict(X_test_mixed)
predictions = result["prediction"].squeeze().numpy()

# Evaluate
accuracy = (predictions == y_test).mean()
print(f"Test Accuracy: {accuracy:.3f}")
```

## Step-by-Step Walkthrough

### 1. Prepare Categorical Features

Categorical features must be **encoded as integers** (0, 1, 2, ...):

```python
from sklearn.preprocessing import LabelEncoder

# Example: Encode product categories
categories = ["Electronics", "Audio", "Electronics", "Audio"]
encoder = LabelEncoder()
categories_encoded = encoder.fit_transform(categories)
# Result: [0, 1, 0, 1]
```

Shape your categorical data as `(n_samples, n_categorical_features)`:

```python
# Single categorical feature
X_categorical = categories_encoded.reshape(-1, 1)

# Multiple categorical features
X_categorical = np.column_stack([
    categories_encoded,
    brands_encoded,
    regions_encoded
])  # Shape: (n_samples, 3)
```

### 2. Configure Categorical Embeddings

Specify vocabulary sizes and embedding dimensions:

```python
model_config = ModelConfig(
    embedding_dim=64,  # For text
    num_classes=2,
    categorical_vocabulary_sizes=[10, 5, 20],  # Vocab size for each feature
    categorical_embedding_dims=[8, 4, 16]       # Embedding dim for each feature
)
```

**Rule of thumb for embedding dimensions:**
```python
embedding_dim = min(50, vocabulary_size // 2)
```

Examples:
- 10 categories → embedding_dim = 5
- 100 categories → embedding_dim = 50
- 1000 categories → embedding_dim = 50 (capped)

### 3. Combine Features

Stack text and categorical data:

```python
# For training
X_train_mixed = np.column_stack([X_text_train, X_cat_train])

# For prediction
X_test_mixed = np.column_stack([X_text_test, X_cat_test])
```

The framework automatically separates text (first column) from categorical features (remaining columns).

### 4. Train and Predict

Training and prediction work the same way:

```python
# Train
classifier.train(X_train_mixed, y_train, training_config=training_config)

# Predict
result = classifier.predict(X_test_mixed)
```

## Comparison: Text-Only vs. Mixed Features

Let's compare performance:

```python
# Text-only model
model_config_text_only = ModelConfig(
    embedding_dim=64,
    num_classes=2
)

classifier_text_only = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config_text_only
)

classifier_text_only.train(X_text_train, y_train, training_config=training_config)
result_text_only = classifier_text_only.predict(X_text_test)
accuracy_text_only = (result_text_only["prediction"].squeeze().numpy() == y_test).mean()

# Mixed features model (from above)
accuracy_mixed = (predictions == y_test).mean()

print(f"Text-Only Accuracy: {accuracy_text_only:.3f}")
print(f"Mixed Features Accuracy: {accuracy_mixed:.3f}")
print(f"Improvement: {(accuracy_mixed - accuracy_text_only):+.3f}")
```

## Combination Strategies

The framework offers different ways to combine categorical embeddings:

### AVERAGE_AND_CONCAT (Default)

Average all categorical embeddings, then concatenate with text:

```python
from torchTextClassifiers.model.components import CategoricalForwardType

model_config = ModelConfig(
    embedding_dim=64,
    num_classes=2,
    categorical_vocabulary_sizes=[10, 5],
    categorical_embedding_dims=[8, 4],
    categorical_forward_type=CategoricalForwardType.AVERAGE_AND_CONCAT
)
```

**Output size:** `text_embedding_dim + avg(categorical_embedding_dims)`

### CONCATENATE_ALL

Concatenate each categorical embedding separately:

```python
model_config = ModelConfig(
    # ... same as above ...
    categorical_forward_type=CategoricalForwardType.CONCATENATE_ALL
)
```

**Output size:** `text_embedding_dim + sum(categorical_embedding_dims)`

**When to use:** Each categorical variable has unique importance.

### SUM_TO_TEXT

Sum all categorical embeddings first:

```python
model_config = ModelConfig(
    # ... same as above ...
    categorical_forward_type=CategoricalForwardType.SUM_TO_TEXT
)
```

**Output size:** `text_embedding_dim + categorical_embedding_dim`

**When to use:** To minimize model size.

## Real-World Example: AG News with Source

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load AG News dataset
df = pd.read_parquet("path/to/ag_news.parquet")
df = df.sample(10000, random_state=42)

# Combine title and description
df['text'] = df['title'] + ' ' + df['description']

# Encode news source as categorical feature
source_encoder = LabelEncoder()
df['source_encoded'] = source_encoder.fit_transform(df['source'])

# Prepare data
X_text = df['text'].values
X_categorical = df['source_encoded'].values.reshape(-1, 1)
y_encoded = LabelEncoder().fit_transform(df['category'])

# Split data
X_text_train, X_text_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
    X_text, X_categorical, y_encoded, test_size=0.2, random_state=42
)

# Train model
tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(X_text_train.tolist())

n_sources = len(source_encoder.classes_)
n_categories = len(np.unique(y_encoded))

model_config = ModelConfig(
    embedding_dim=128,
    num_classes=n_categories,
    categorical_vocabulary_sizes=[n_sources],
    categorical_embedding_dims=[min(50, n_sources // 2)]
)

classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

X_train_mixed = np.column_stack([X_text_train, X_cat_train])
X_test_mixed = np.column_stack([X_text_test, X_cat_test])

training_config = TrainingConfig(
    num_epochs=50,
    batch_size=128,
    lr=1e-3,
    patience_early_stopping=3
)

classifier.train(X_train_mixed, y_train, training_config=training_config)

# Evaluate
result = classifier.predict(X_test_mixed)
accuracy = (result["prediction"].squeeze().numpy() == y_test).mean()
print(f"Test Accuracy: {accuracy:.3f}")
```

## Common Issues

### Issue 1: Shape Mismatch

**Error:** "Expected 2D array, got 1D array"

**Solution:** Reshape single features:
```python
X_categorical = categories.reshape(-1, 1)  # Add column dimension
```

### Issue 2: Non-Integer Categories

**Error:** "Expected integer values"

**Solution:** Use `LabelEncoder`:
```python
encoder = LabelEncoder()
categories_encoded = encoder.fit_transform(categories)
```

### Issue 3: Missing Vocabulary Sizes

**Error:** "Must specify categorical_vocabulary_sizes"

**Solution:** Provide vocab size for each categorical feature:
```python
vocab_sizes = [int(np.max(X_cat_train[:, i]) + 1) for i in range(X_cat_train.shape[1])]
model_config = ModelConfig(
    categorical_vocabulary_sizes=vocab_sizes,
    categorical_embedding_dims=[min(50, v // 2) for v in vocab_sizes]
)
```

### Issue 4: No Performance Improvement

**Possible reasons:**
- Categorical features not predictive of labels
- Text already contains categorical information
- Need more training data
- Categorical embeddings too small

**Try:**
- Increase embedding dimensions
- Check feature-label correlation
- Try different combination strategies

## Customization

### Custom Embedding Dimensions

Different dimensions for different importance:

```python
model_config = ModelConfig(
    embedding_dim=128,
    num_classes=5,
    categorical_vocabulary_sizes=[100, 10, 50],
    categorical_embedding_dims=[32, 4, 16]  # Vary by importance
)
```

### With Attention

Combine categorical features with attention-based text embeddings:

```python
from torchTextClassifiers.model.components import AttentionConfig

attention_config = AttentionConfig(
    n_embd=128,
    n_head=8,
    n_layer=3
)

model_config = ModelConfig(
    embedding_dim=128,
    num_classes=5,
    attention_config=attention_config,
    categorical_vocabulary_sizes=[100],
    categorical_embedding_dims=[32]
)
```

## Best Practices

1. **Start simple:** Begin with text-only model, add categorical features if needed
2. **Check correlation:** Ensure categorical features correlate with labels
3. **Normalize vocabulary sizes:** Use embedding_dim ≈ vocabulary_size // 2
4. **Avoid overfitting:** Don't use too many high-dimensional categorical features
5. **Compare performance:** Always compare mixed vs. text-only models

## Next Steps

- **Explainability**: Learn which features (text or categorical) drive predictions in {doc}`explainability`
- **Multilabel**: Apply mixed features to multilabel tasks in {doc}`multilabel_classification`
- **Advanced Training**: Explore hyperparameter tuning with mixed features

## Summary

**Key takeaways:**
- Categorical features can improve classification performance
- Encode categories as integers (0, 1, 2, ...)
- Configure vocabulary sizes and embedding dimensions
- Combine text and categorical data using `np.column_stack`
- Compare performance to validate improvement

Ready to understand your model's predictions? Continue to {doc}`explainability`!
