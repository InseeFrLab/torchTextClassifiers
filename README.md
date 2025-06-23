# torchTextClassifiers

A unified, extensible framework for text classification using PyTorch and PyTorch Lightning.

## 🚀 Features

- **Unified API**: Consistent interface for different classifier types
- **FastText Support**: Built-in FastText classifier implementation
- **PyTorch Lightning**: Automated training with callbacks, early stopping, and logging
- **Mixed Features**: Support for both text and categorical features
- **Extensible**: Easy to add new classifier types
- **Production Ready**: Model serialization, validation, and inference

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/torch-fastText.git
cd torch-fastText

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

### Working with Mixed Features (Text + Categorical)

```python
import numpy as np
from torchTextClassifiers import create_fasttext

# Text data with categorical features
X_train = np.column_stack([
    np.array(["Great product!", "Terrible service", "Love it!"]),  # Text
    np.array([[1, 2], [2, 1], [1, 3]])  # Categorical features
])
y_train = np.array([1, 0, 1])

# Create classifier with categorical support
classifier = create_fasttext(
    embedding_dim=50,
    sparse=False,
    num_tokens=5000,
    min_count=1,
    min_n=3,
    max_n=6,
    len_word_ngrams=2,
    num_classes=2,
    categorical_vocabulary_sizes=[3, 4],  # Vocab sizes for categorical features
    categorical_embedding_dims=[10, 10]   # Embedding dims for categorical features
)

# Build and train as usual
classifier.build(X_train, y_train)
# ... continue with training
```

### Model Persistence

```python
# Save configuration
classifier.to_json('model_config.json')

# Load configuration (creates new instance)
new_classifier = torchTextClassifiers.from_json('model_config.json')

# You'll need to retrain the loaded classifier
new_classifier.build(X_train, y_train)
new_classifier.train(X_train, y_train, X_val, y_val, ...)
```

## 🔧 Advanced Usage

### Custom Configuration

```python
from torchTextClassifiers import torchTextClassifiers, ClassifierType
from torchTextClassifiers.classifiers.fasttext.config import FastTextConfig

# Create custom configuration
config = FastTextConfig(
    embedding_dim=200,
    sparse=True,
    num_tokens=20000,
    min_count=3,
    min_n=2,
    max_n=8,
    len_word_ngrams=3,
    num_classes=5,
    direct_bagging=False,  # Custom FastText parameter
)

# Create classifier with custom config
classifier = torchTextClassifiers(ClassifierType.FASTTEXT, config)
```

### Using Pre-trained Tokenizers

```python
from torchTextClassifiers import build_fasttext_from_tokenizer

# Assume you have a pre-trained tokenizer
# my_tokenizer = ... (previously trained NGramTokenizer)

classifier = build_fasttext_from_tokenizer(
    tokenizer=my_tokenizer,
    embedding_dim=100,
    num_classes=3,
    sparse=False
)

# Model and tokenizer are already built, ready for training
classifier.train(X_train, y_train, X_val, y_val, ...)
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

## 📊 API Reference

### Main Classes

#### `torchTextClassifiers`
The main classifier class providing a unified interface.

**Key Methods:**
- `build(X_train, y_train)`: Build tokenizer and model
- `train(X_train, y_train, X_val, y_val, ...)`: Train the model
- `predict(X)`: Make predictions
- `validate(X, Y)`: Evaluate on test data
- `to_json(filepath)`: Save configuration
- `from_json(filepath)`: Load configuration

#### `ClassifierType`
Enumeration of supported classifier types.
- `FASTTEXT`: FastText classifier

#### `ClassifierFactory`
Factory for creating classifier instances.

### FastText Specific

#### `create_fasttext(**kwargs)`
Convenience function to create FastText classifiers.

**Parameters:**
- `embedding_dim`: Embedding dimension
- `sparse`: Use sparse embeddings
- `num_tokens`: Vocabulary size
- `min_count`: Minimum token frequency
- `min_n`, `max_n`: Character n-gram range
- `len_word_ngrams`: Word n-gram length
- `num_classes`: Number of output classes

#### `build_fasttext_from_tokenizer(tokenizer, **kwargs)`
Create FastText classifier from existing tokenizer.

## 🏗️ Architecture

The framework follows a modular architecture:

```
torchTextClassifiers/
├── torchTextClassifiers.py      # Main classifier interface
├── classifiers/
│   ├── base.py                  # Abstract base classes
│   └── fasttext/                # FastText implementation
│       ├── config.py            # Configuration
│       ├── wrapper.py           # Classifier wrapper
│       ├── factory.py           # Convenience methods
│       ├── tokenizer.py         # N-gram tokenizer
│       ├── pytorch_model.py     # PyTorch model
│       ├── lightning_module.py  # Lightning module
│       └── dataset.py           # Dataset implementation
├── utilities/
│   └── checkers.py              # Input validation utilities
└── factories.py                 # Generic factory system
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

## 🤝 Contributing

We welcome contributions! See our [Developer Guide](docs/developer_guide.md) for information on:

- Adding new classifier types
- Code organization and patterns
- Testing requirements
- Documentation standards

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Inspired by [FastText](https://fasttext.cc/) for efficient text classification
- Uses [uv](https://github.com/astral-sh/uv) for dependency management

## 📚 Examples

See the [examples/](examples/) directory for:
- Basic text classification
- Multi-class classification
- Mixed features (text + categorical)
- Custom classifier implementation
- Advanced training configurations

## 🐛 Support

If you encounter any issues:

1. Check the [examples](examples/) for similar use cases
2. Review the API documentation above
3. Open an issue on GitHub with:
   - Python version
   - Package versions (`uv tree` or `pip list`)
   - Minimal reproduction code
   - Error messages/stack traces