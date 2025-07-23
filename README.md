# torchTextClassifiers

A unified, extensible framework for text classification built on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## 🚀 Features

- **Unified API**: Consistent interface for different classifier wrappers
- **Extensible**: Easy to add new classifier implementations through wrapper pattern
- **FastText Support**: Built-in FastText classifier with n-gram tokenization
- **Flexible Preprocessing**: Each classifier can implement its own text preprocessing approach
- **PyTorch Lightning**: Automated training with callbacks, early stopping, and logging


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/InseeFrLab/torchTextClassifiers.git
cd torchtextClassifiers

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## 🎯 Quick Start

### Basic FastText Classification

```python
import numpy as np
from torchTextClassifiers import create_fasttext

# Create a FastText classifier
classifier = create_fasttext(
    embedding_dim=100,
    sparse=False,
    num_tokens=10000,
    min_count=2,
    min_n=3,
    max_n=6,
    len_word_ngrams=2,
    num_classes=2
)

# Prepare your data
X_train = np.array([
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative case"
])
y_train = np.array([1, 0, 1, 0])

X_val = np.array([
    "Validation positive",
    "Validation negative"
])
y_val = np.array([1, 0])

# Build the model
classifier.build(X_train, y_train)

# Train the model
classifier.train(
    X_train, y_train, X_val, y_val,
    num_epochs=50,
    batch_size=32,
    patience_train=5,
    verbose=True
)

# Make predictions
X_test = np.array(["This is a test sentence"])
predictions = classifier.predict(X_test)
print(f"Predictions: {predictions}")

# Validate on test set
accuracy = classifier.validate(X_test, np.array([1]))
print(f"Accuracy: {accuracy:.3f}")
```

### Custom Classifier Implementation

```python
import numpy as np
from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.classifiers.simple_text_classifier import SimpleTextWrapper, SimpleTextConfig

# Example: TF-IDF based classifier (alternative to tokenization)
config = SimpleTextConfig(
    hidden_dim=128,
    num_classes=2,
    max_features=5000,
    learning_rate=1e-3,
    dropout_rate=0.2
)

# Create classifier with TF-IDF preprocessing
wrapper = SimpleTextWrapper(config)
classifier = torchTextClassifiers(wrapper)

# Text data
X_train = np.array(["Great product!", "Terrible service", "Love it!"])
y_train = np.array([1, 0, 1])

# Build and train
classifier.build(X_train, y_train)
# ... continue with training
```


### Training Customization

```python
# Custom PyTorch Lightning trainer parameters
trainer_params = {
    'accelerator': 'gpu',
    'devices': 1,
    'precision': 16,  # Mixed precision training
    'gradient_clip_val': 1.0,
}

classifier.train(
    X_train, y_train, X_val, y_val,
    num_epochs=100,
    batch_size=64,
    patience_train=10,
    trainer_params=trainer_params,
    verbose=True
)
```

## 🔬 Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=torchTextClassifiers

# Run specific test file
uv run pytest tests/test_torchTextClassifiers.py -v
```


## 📚 Examples

See the [examples/](examples/) directory for:
- Basic text classification
- Multi-class classification
- Mixed features (text + categorical)
- Custom classifier implementation
- Advanced training configurations



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
