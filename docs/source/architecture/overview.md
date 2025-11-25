# Architecture Overview

torchTextClassifiers is a **modular, component-based framework** for text classification. Rather than a black box, it provides clear, reusable components that you can understand, configure, and compose.

## The Pipeline

At its core, torch Text Classifiers processes data through a simple pipeline:

```{mermaid}
flowchart LR
    TextInput["Text Input"] --> Tokenizer
    Tokenizer --> TextEmbedder["Text Embedder"]

    CatInput["Categorical Features<br/>(Optional)"] --> CatEmbedder["Categorical Embedder"]

    TextEmbedder --> ClassHead["Classification Head"]
    CatEmbedder --> ClassHead

    ClassHead --> Predictions

    style CatInput stroke-dasharray: 5 5
    style CatEmbedder stroke-dasharray: 5 5
```

**Data Flow:**
1. **Text** is tokenized into numerical tokens
2. **Tokens** are embedded into dense vectors (with optional attention)
3. **Categorical variables** (optional) are embedded separately
4. **All embeddings** are combined
5. **Classification head** produces final predictions

## Component 1: Tokenizer

**Purpose:** Convert text strings into numerical tokens that the model can process.

### Available Tokenizers

torchTextClassifiers supports three tokenization strategies:

#### NGramTokenizer (FastText-style)

Character n-gram tokenization for robustness to typos and rare words.

```python
from torchTextClassifiers.tokenizers import NGramTokenizer

tokenizer = NGramTokenizer(
    vocab_size=10000,
    min_n=3,  # Minimum n-gram size
    max_n=6,  # Maximum n-gram size
)
tokenizer.train(training_texts)
```

**When to use:**
- Text with typos or non-standard spellings
- Morphologically rich languages
- Limited training data

#### WordPieceTokenizer

Subword tokenization for balanced vocabulary coverage.

```python
from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(training_texts)
```

**When to use:**
- Standard text classification
- Moderate vocabulary size
- Good balance of coverage and granularity

#### HuggingFaceTokenizer

Use pre-trained tokenizers from HuggingFace.

```python
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)
```

**When to use:**
- Transfer learning from pre-trained models
- Need specific language support
- Want to leverage existing tokenizers

### Tokenizer Output

All tokenizers produce the same output format:

```python
output = tokenizer(["Hello world!", "Text classification"])
# output.input_ids: Token indices (batch_size, seq_len)
# output.attention_mask: Attention mask (batch_size, seq_len)
```

## Component 2: Text Embedder

**Purpose:** Convert tokens into dense, semantic embeddings that capture meaning.

### Basic Text Embedding

```python
from torchTextClassifiers.model.components import TextEmbedder, TextEmbedderConfig

config = TextEmbedderConfig(
    vocab_size=5000,
    embedding_dim=128,
)
embedder = TextEmbedder(config)

# Forward pass
text_features = embedder(token_ids)  # Shape: (batch_size, 128)
```

**How it works:**
1. Looks up embedding for each token
2. Averages embeddings across the sequence
3. Produces a fixed-size vector per sample

### With Self-Attention (Optional)

Add transformer-style self-attention for better contextual understanding:

```python
from torchTextClassifiers.model.components import AttentionConfig

attention_config = AttentionConfig(
    n_embd=128,
    n_head=4,      # Number of attention heads
    n_layer=2,     # Number of transformer blocks
    dropout=0.1,
)

config = TextEmbedderConfig(
    vocab_size=5000,
    embedding_dim=128,
    attention_config=attention_config,  # Add attention
)
embedder = TextEmbedder(config)
```

**When to use attention:**
- Long documents where context matters
- Tasks requiring understanding of word relationships
- When you have sufficient training data

**Configuration:**
- `embedding_dim`: Size of embedding vectors (e.g., 64, 128, 256)
- `n_head`: Number of attention heads (typically 4, 8, or 16)
- `n_layer`: Depth of transformer (start with 2-3)

## Component 3: Categorical Variable Handler

**Purpose:** Process categorical features (like user demographics, product categories) alongside text.

### When to Use

Add categorical features when you have structured data that complements text:
- User age, location, or demographics
- Product categories or attributes
- Document metadata (source, type, etc.)

### Setup

```python
from torchTextClassifiers.model.components import (
    CategoricalVariableNet,
    CategoricalForwardType
)

# Example: 3 categorical variables
# - Variable 1: 10 possible values
# - Variable 2: 5 possible values
# - Variable 3: 20 possible values

cat_handler = CategoricalVariableNet(
    vocabulary_sizes=[10, 5, 20],
    embedding_dims=[8, 4, 16],    # Embedding size for each
    forward_type=CategoricalForwardType.AVERAGE_AND_CONCAT
)
```

### Combination Strategies

The `forward_type` controls how categorical embeddings are combined:

#### AVERAGE_AND_CONCAT

Average all categorical embeddings, then concatenate with text:

![Average and Concatenate](diagrams/avg_concat.png)

```python
forward_type=CategoricalForwardType.AVERAGE_AND_CONCAT
```

**Output size:** `text_embedding_dim + sum(categorical_embedding_dims)/n_categoricals`

**When to use:** When categorical variables are equally important

#### CONCATENATE_ALL

Concatenate each categorical embedding separately:

![Full Concatenation](diagrams/full_concat.png)

```python
forward_type=CategoricalForwardType.CONCATENATE_ALL
```

**Output size:** `text_embedding_dim + sum(categorical_embedding_dims)`

**When to use:** When each categorical variable has unique importance

#### SUM_TO_TEXT

Sum all categorical embeddings, then concatenate:

```python
forward_type=CategoricalForwardType.SUM_TO_TEXT
```

**Output size:** `text_embedding_dim + categorical_embedding_dim`

**When to use:** To minimize output dimension

### Example with Data

```python
# Text data
texts = ["Sample 1", "Sample 2"]

# Categorical data: (n_samples, n_categorical_variables)
categorical = np.array([
    [5, 2, 14],  # Sample 1: cat1=5, cat2=2, cat3=14
    [3, 1, 8],   # Sample 2: cat1=3, cat2=1, cat3=8
])

# Process
cat_features = cat_handler(categorical)  # Shape: (2, total_emb_dim)
```

## Component 4: Classification Head

**Purpose:** Take the combined features and produce class predictions.

### Simple Classification

```python
from torchTextClassifiers.model.components import ClassificationHead

head = ClassificationHead(
    input_dim=152,      # 128 (text) + 24 (categorical)
    num_classes=5,      # Number of output classes
)

logits = head(combined_features)  # Shape: (batch_size, 5)
```

### Custom Classification Head

For more complex classification, provide your own architecture:

```python
import torch.nn as nn

custom_head = nn.Sequential(
    nn.Linear(152, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 5)
)

head = ClassificationHead(linear=custom_head)
```

## Complete Architecture

![Complete Architecture](diagrams/NN.drawio.png)
*Complete model architecture showing all components*

### Full Model Assembly

The framework automatically combines all components:

```python
from torchTextClassifiers.model import TextClassificationModel

model = TextClassificationModel(
    text_embedder=text_embedder,
    categorical_variable_net=cat_handler,  # Optional
    classification_head=head,
)

# Forward pass
logits = model(token_ids, categorical_data)
```

## Usage Examples

### Example 1: Text-Only Classification

Simple sentiment analysis with just text:

```python
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# 1. Create tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000)
tokenizer.train(texts)

# 2. Configure model
model_config = ModelConfig(
    embedding_dim=128,
    num_classes=2,  # Binary classification
)

# 3. Train
classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)
training_config = TrainingConfig(num_epochs=10, batch_size=32, lr=1e-3)
classifier.train(texts, labels, training_config=training_config)

# 4. Predict
predictions = classifier.predict(new_texts)
```

### Example 2: Mixed Features (Text + Categorical)

Product classification using both description and category:

```python
import numpy as np

# Text + categorical data
texts = ["Product description...", "Another product..."]
categorical = np.array([
    [3, 1],  # Product 1: category=3, brand=1
    [5, 2],  # Product 2: category=5, brand=2
])
labels = [0, 1]

# Configure model with categorical features
model_config = ModelConfig(
    embedding_dim=128,
    num_classes=3,
    categorical_vocabulary_sizes=[10, 5],  # 10 categories, 5 brands
    categorical_embedding_dims=[8, 4],
)

# Train
classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)
classifier.train(
    X_text=texts,
    y=labels,
    X_categorical=categorical,
    training_config=training_config
)
```

### Example 3: With Attention

For longer documents or complex text:

```python
from torchTextClassifiers.model.components import AttentionConfig

# Add attention for better understanding
attention_config = AttentionConfig(
    n_embd=128,
    n_head=8,
    n_layer=3,
    dropout=0.1,
)

model_config = ModelConfig(
    embedding_dim=128,
    num_classes=5,
    attention_config=attention_config,  # Enable attention
)

classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)
```

### Example 4: Custom Components

For maximum flexibility, compose components manually:

```python
from torch import nn
from torchTextClassifiers.model.components import TextEmbedder, ClassificationHead

# Create custom model
class CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedder = TextEmbedder(text_config)
        self.custom_layer = nn.Linear(128, 64)
        self.head = ClassificationHead(64, num_classes)

    def forward(self, input_ids):
        text_features = self.text_embedder(input_ids)
        custom_features = self.custom_layer(text_features)
        return self.head(custom_features)
```

## Using the High-Level API

For most users, the `torchTextClassifiers` wrapper handles all the complexity:

```python
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig

# Simple 3-step process:
# 1. Create tokenizer and train it
# 2. Configure model architecture
# 3. Train and predict

classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)
classifier.train(texts, labels, training_config=training_config)
predictions = classifier.predict(new_texts)
```

**What the wrapper does:**
- Creates all components automatically
- Sets up PyTorch Lightning training
- Handles data loading and batching
- Provides simple train/predict interface
- Manages configurations

**When to use the wrapper:**
- Standard classification tasks
- Quick experimentation
- Don't need custom architecture
- Want simplicity over control

## For Advanced Users

### Direct PyTorch Usage

All components are standard `torch.nn.Module` objects:

```python
# All components work with standard PyTorch
isinstance(text_embedder, nn.Module)  # True
isinstance(cat_handler, nn.Module)  # True
isinstance(head, nn.Module)  # True

# Use in any PyTorch code
model = TextClassificationModel(text_embedder, cat_handler, head)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Standard PyTorch training loop
for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch.input_ids, batch.categorical)
    loss = criterion(logits, batch.labels)
    loss.backward()
    optimizer.step()
```

### PyTorch Lightning Integration

For automated training with advanced features:

```python
from torchTextClassifiers.model import TextClassificationModule
from pytorch_lightning import Trainer

# Wrap model in Lightning module
lightning_module = TextClassificationModule(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    lr=1e-3,
)

# Use Lightning Trainer
trainer = Trainer(
    max_epochs=20,
    accelerator="gpu",
    devices=4,  # Multi-GPU
    callbacks=[EarlyStopping(), ModelCheckpoint()],
)
trainer.fit(lightning_module, train_dataloader, val_dataloader)
```

## Design Philosophy

### Modularity

Each component is independent and can be used separately:

```python
# Use just the tokenizer
tokenizer = NGramTokenizer()

# Use just the embedder
embedder = TextEmbedder(config)

# Use just the classifier head
head = ClassificationHead(input_dim, num_classes)
```

### Flexibility

Mix and match components for your use case:

```python
# Text only
model = TextClassificationModel(text_embedder, None, head)

# Text + categorical
model = TextClassificationModel(text_embedder, cat_handler, head)

# Custom combination
model = MyCustomModel(text_embedder, my_layer, head)
```

### Simplicity

Sensible defaults for quick starts:

```python
# Minimal configuration
model_config = ModelConfig(embedding_dim=128, num_classes=2)

# Or detailed configuration
model_config = ModelConfig(
    embedding_dim=256,
    num_classes=10,
    categorical_vocabulary_sizes=[50, 20, 100],
    categorical_embedding_dims=[32, 16, 64],
    attention_config=AttentionConfig(n_embd=256, n_head=8, n_layer=4),
)
```

### Extensibility

Easy to add custom components:

```python
class MyCustomEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom implementation

    def forward(self, input_ids):
        # Your custom forward pass
        return embeddings

# Use with existing components
model = TextClassificationModel(
    text_embedder=MyCustomEmbedder(),
    classification_head=head,
)
```

## Configuration Guide

### Choosing Embedding Dimension

| Task Complexity | Data Size | Recommended embedding_dim |
|----------------|-----------|---------------------------|
| Simple (binary) | < 1K samples | 32-64 |
| Medium (3-5 classes) | 1K-10K samples | 64-128 |
| Complex (10+ classes) | 10K-100K samples | 128-256 |
| Very complex | > 100K samples | 256-512 |

### Attention Configuration

| Document Length | Recommended Setup |
|----------------|-------------------|
| Short (< 50 tokens) | No attention needed |
| Medium (50-200 tokens) | n_layer=2, n_head=4 |
| Long (200-512 tokens) | n_layer=3-4, n_head=8 |
| Very long (> 512 tokens) | n_layer=4-6, n_head=8-16 |

### Categorical Embedding Size

Rule of thumb: `embedding_dim ≈ min(50, vocabulary_size // 2)`

```python
# For categorical variable with 100 unique values:
categorical_embedding_dim = min(50, 100 // 2) = 50

# For categorical variable with 10 unique values:
categorical_embedding_dim = min(50, 10 // 2) = 5
```

## Summary

torchTextClassifiers provides a **component-based pipeline** for text classification:

1. **Tokenizer** → Converts text to tokens
2. **Text Embedder** → Creates semantic embeddings (with optional attention)
3. **Categorical Handler** → Processes additional features (optional)
4. **Classification Head** → Produces predictions

**Key Benefits:**
- Clear data flow through intuitive components
- Mix and match for your specific needs
- Start simple, add complexity as needed
- Full PyTorch compatibility

## Next Steps

- **Tutorials**: See {doc}`../tutorials/index` for step-by-step guides
- **API Reference**: Check {doc}`../api/index` for detailed documentation
- **Examples**: Explore complete examples in the repository

Ready to build your classifier? Start with {doc}`../getting_started/quickstart`!
