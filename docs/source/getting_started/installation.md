# Installation

## Requirements

torchTextClassifiers requires:

- **Python**: 3.11 or higher
- **PyTorch**: Will be installed automatically as a dependency via pytorch-lightning
- **Operating System**: Linux, macOS, or Windows

## Installation from Source

Currently, torchTextClassifiers is available only from source. Clone the repository and install using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

```bash
# Clone the repository
git clone https://github.com/InseeFrLab/torchTextClassifiers.git
cd torchTextClassifiers

# Install with uv
uv sync
```

## Optional Dependencies

torchTextClassifiers comes with optional dependency groups for additional features:

### Explainability Support

For model interpretation and explainability features:

```bash
uv sync --extra explainability
```

This installs:
- `captum`: For attribution analysis
- `nltk`: For text preprocessing
- `unidecode`: For text normalization

### HuggingFace Integration

To use HuggingFace tokenizers:

```bash
uv sync --extra huggingface
```

This installs:
- `tokenizers`: Fast tokenizers
- `transformers`: HuggingFace transformers
- `datasets`: HuggingFace datasets

### Text Preprocessing

For additional text preprocessing utilities:

```bash
uv sync --extra preprocess
```

This installs:
- `nltk`: Natural language toolkit
- `unidecode`: Text normalization

### All Optional Dependencies

Install all extras at once:

```bash
uv sync --all-extras
```

### Development Dependencies

If you want to contribute to the project:

```bash
uv sync --group dev
```

## Verification

Verify your installation by running:

```python
import torchTextClassifiers
print(torchTextClassifiers.__version__)  # Should print: 0.0.0-dev
```

Or try a simple import:

```python
from torchTextClassifiers import torchTextClassifiers, ModelConfig, TrainingConfig
from torchTextClassifiers.tokenizers import WordPieceTokenizer

print("Installation successful!")
```

## GPU Support

torchTextClassifiers uses PyTorch Lightning, which automatically detects and uses GPUs if available.

To use GPUs, make sure you have:
1. CUDA-compatible GPU
2. CUDA toolkit installed
3. PyTorch installed with CUDA support

Check GPU availability:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you've installed the package:

```bash
# Reinstall
uv sync
```

### Dependency Conflicts

If you have dependency conflicts, try creating a fresh virtual environment:

```bash
# Create new virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### PyTorch Installation Issues

If PyTorch installation fails, uv will handle it automatically through pytorch-lightning. If you need a specific PyTorch version, you can specify it in your environment before running:

```bash
# For CPU-only PyTorch
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
uv sync

# For GPU (CUDA 11.8)
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
uv sync
```

## Next Steps

Now that you have torchTextClassifiers installed, head over to the {doc}`quickstart` to build your first classifier!
