# torchTextClassifiers

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://inseefrlab.github.io/torchTextClassifiers/)

A unified, extensible framework for text classification with categorical variables built on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## 🚀 Features

- **Complex input support**: Handle text data alongside categorical variables seamlessly.
  - **ValueEncoder**: Pass raw string categorical values and labels directly — no manual integer encoding required. Build a `ValueEncoder` from `DictEncoder` or sklearn `LabelEncoder` instances once, and the wrapper handles encoding at train time and label decoding after prediction automatically.
- **Unified yet highly customizable**:
    - Use any tokenizer from HuggingFace or the original fastText's ngram tokenizer.
    - Text embedding is split into two composable stages: **`TokenEmbedder`** (token → per-token vectors, with optional self-attention) and **`SentenceEmbedder`** (aggregation: mean / first / last / label attention). Combine them with `CategoricalVariableNet` and `ClassificationHead` — all are `torch.nn.Module`.
    - The `TextClassificationModel` class assembles these components and can be extended for custom behavior.
- **Multiclass / multilabel classification support**: Support for both multiclass (only one label is true) and multi-label (several labels can be true) classification tasks.
- **PyTorch Lightning**: Automated training with callbacks, early stopping, and logging
- **Easy experimentation**: Simple API for training, evaluating, and predicting with minimal code:
    - The `torchTextClassifiers` wrapper class orchestrates the tokenizer and the model for you
- **Explainability**:
    - **Captum integration**: gradient-based token attribution via integrated gradients (`explain_with_captum=True`).
    - **Label attention**: class-specific cross-attention that produces one sentence embedding per class, enabling token-level explanations for each label (`explain_with_label_attention=True`). Enable it by setting `n_heads_label_attention` in `ModelConfig`.


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/InseeFrLab/torchTextClassifiers.git
cd torchTextClassifiers

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
