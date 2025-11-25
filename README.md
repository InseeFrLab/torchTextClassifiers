# torchTextClassifiers

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://inseefrlab.github.io/torchTextClassifiers/)

A unified, extensible framework for text classification with categorical variables built on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## 🚀 Features

- **Complex input support**: Handle text data alongside categorical variables seamlessly.
- **Unified yet highly customizable**:
    - Use any tokenizer from HuggingFace or the original fastText's ngram tokenizer.
    - Manipulate the components (`TextEmbedder`, `CategoricalVariableNet`, `ClassificationHead`) to easily create custom architectures - including **self-attention**. All of them are `torch.nn.Module` !
    - The `TextClassificationModel` class combines these components and can be extended for custom behavior.
- **Multiclass / multilabel classification support**: Support for both multiclass (only one label is true) and multi-label (several labels can be true) classification tasks.
- **PyTorch Lightning**: Automated training with callbacks, early stopping, and logging
- **Easy experimentation**: Simple API for training, evaluating, and predicting with minimal code:
    - The `torchTextClassifiers` wrapper class orchestrates the tokenizer and the model for you
- **Additional features**: explainability using Captum


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

## 📖 Documentation

Full documentation is available at: **https://inseefrlab.github.io/torchTextClassifiers/**
The documentation includes:
- **Getting Started**: Installation and quick start guide
- **Architecture**: Understanding the 3-layer design
- **Tutorials**: Step-by-step guides for different use cases
- **API Reference**: Complete API documentation

## 📝 Usage

Checkout the [notebook](notebooks/example.ipynb) for a quick start.

## 📚 Examples

See the [examples/](examples/) directory for:
- Basic text classification
- Multi-class classification
- Mixed features (text + categorical)
- Advanced training configurations
- Prediction and explainability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


