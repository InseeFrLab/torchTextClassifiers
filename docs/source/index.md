# torchTextClassifiers

**A unified, extensible framework for text classification with categorical variables built on PyTorch and PyTorch Lightning.**

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
architecture/index
tutorials/index
api/index
```

## Welcome

torchTextClassifiers is a Python package designed to simplify building, training, and evaluating deep learning text classifiers. Whether you're working on sentiment analysis, document categorization, or any text classification task, this framework provides the tools you need while maintaining flexibility for customization.

## Key Features

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Complex Input Support
:text-align: center

Handle text data alongside categorical variables seamlessly
:::

:::{grid-item-card} Highly Customizable
:text-align: center

Use any tokenizer from HuggingFace or FastText's n-gram tokenizer
:::

:::{grid-item-card} Multiclass & Multilabel
:text-align: center

Support for both multiclass and multi-label classification tasks
:::

:::{grid-item-card} PyTorch Lightning
:text-align: center

Automated training with callbacks, early stopping, and logging
:::

:::{grid-item-card} Modular Architecture
:text-align: center

Mix and match components to create custom architectures
:::

:::{grid-item-card} Built-in Explainability
:text-align: center

Understand predictions using Captum integration
:::

::::

## Quick Example

Here's a minimal example to get you started:

```python
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Sample data
texts = ["I love this product!", "Terrible experience", "It's okay"]
labels = [1, 0, 1]  # Binary classification

# Create and train tokenizer
tokenizer = WordPieceTokenizer()
tokenizer.train(texts, vocab_size=1000)

# Configure model
model_config = ModelConfig(embedding_dim=64, num_classes=2)
training_config = TrainingConfig(num_epochs=5, batch_size=16, lr=1e-3)

# Create classifier
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)

# Train
classifier.train(texts, labels, training_config=training_config)

# Predict
predictions = classifier.predict(["Best product ever!"])
```

## Installation

Currently, install from source:

```bash
# Clone the repository
git clone https://github.com/InseeFrLab/torchTextClassifiers.git
cd torchTextClassifiers

# Install with uv (recommended)
uv sync
```

### Optional Dependencies

Install additional features as needed:

```bash
# For explainability features
uv sync --extra explainability

# For HuggingFace tokenizers
uv sync --extra huggingface

# For text preprocessing
uv sync --extra preprocess

# Install all extras
uv sync --all-extras
```

## Get Started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`rocket` Quick Start
:link: getting_started/quickstart
:link-type: doc

Get up and running in 5 minutes with a complete working example
:::

:::{grid-item-card} {fas}`layer-group` Architecture
:link: architecture/overview
:link-type: doc

Understand the component-based pipeline and design philosophy
:::

:::{grid-item-card} {fas}`graduation-cap` Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step guides for different use cases and features
:::

:::{grid-item-card} {fas}`book` API Reference
:link: api/index
:link-type: doc

Complete API documentation for all classes and functions
:::

::::

## Why torchTextClassifiers?

### Unified API

Work with a consistent, simple API whether you're doing binary, multiclass, or multilabel classification. The `torchTextClassifiers` wrapper class handles all the complexity.

### Flexible Components

All components (`TextEmbedder`, `CategoricalVariableNet`, `ClassificationHead`) are standard `torch.nn.Module` objects. Mix and match them or create your own custom components.

### Production Ready

Built on PyTorch Lightning for robust training with automatic:
- Early stopping
- Checkpointing
- Logging
- Multi-GPU support

### Explainability First

Understand what your model is learning with built-in Captum integration for word-level and character-level attribution analysis.

## Use Cases

- **Sentiment Analysis**: Binary or multi-class sentiment classification
- **Document Categorization**: Classify documents into multiple categories
- **Mixed Feature Classification**: Combine text with categorical variables (e.g., user demographics)
- **Multilabel Classification**: Assign multiple labels to each text sample
- **Model Interpretation**: Understand which words contribute to predictions

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/InseeFrLab/torchTextClassifiers/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request on [GitHub](https://github.com/InseeFrLab/torchTextClassifiers).

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/InseeFrLab/torchTextClassifiers/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/InseeFrLab/torchTextClassifiers/discussions)
