# MLflow Logging Example

This example demonstrates how to train a text classifier with torchTextClassifiers
and log it to MLflow with a pyfunc wrapper for flexible deployment.

## Features

- **Training with MLflow Logging**: Uses PyTorch Lightning's `MLFlowLogger` to log
  training metrics (loss, accuracy) per epoch automatically
- **Flexible Input Formats**: The pyfunc wrapper accepts strings, lists, DataFrames
- **Top-K Predictions**: Return multiple predictions per sample with confidence scores
- **Explainability**: Token-level attributions via Captum's Integrated Gradients

## Prerequisites

Install the required dependencies:

```bash
# Using uv (recommended)
uv add torchTextClassifiers mlflow captum

# Or using pip
pip install torchTextClassifiers mlflow captum
```

## Quick Start

Run the example script:

```bash
# From the repository root
uv run examples/mlflow_logging/run_example.py

# Or with explicit dependencies
uv run --extra huggingface --with mlflow --with captum \
    examples/mlflow_logging/run_example.py
```

## Package Structure

```
mlflow_logging/
├── __init__.py           # Package exports (TextClassifierWrapper)
├── pyfunc_wrapper.py     # MLflow pyfunc wrapper class
├── run_example.py        # Main example script
├── README.md             # This file
└── data/
    ├── train.csv         # Training data (30 samples)
    ├── val.csv           # Validation data (6 samples)
    └── test.csv          # Test data (6 samples)
```

## Data Format

The CSV files contain product reviews with sentiment labels:

| Column   | Description                                    |
|----------|------------------------------------------------|
| text     | Product review text                            |
| category | Product category (0=electronics, 1=clothing, 2=books) |
| label    | Sentiment label (0=negative, 1=positive)       |

## Using the Logged Model

After running the example, load and use the model:

```python
import mlflow

# Load the model (replace <run_id> with actual run ID)
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Basic prediction - single string
model.predict("Great product!")

# Multiple samples as list of strings
model.predict(["Love it!", "Terrible quality."])

# With categorical features as list of lists
model.predict([["Amazing electronics!", 0], ["Bad fit.", 1]])

# DataFrame input
import pandas as pd
df = pd.DataFrame({"text": ["Nice book!"], "category": [2]})
model.predict(df)
```

## Inference Parameters

The pyfunc wrapper supports inference-time parameters via the `params` argument:

### `top_k` - Multiple Predictions

Return the top-k predictions with confidence scores:

```python
# Get top 3 predictions per sample
result = model.predict(data, params={"top_k": 3})

# Result columns: prediction_1, confidence_1, prediction_2, confidence_2, ...
print(result)
#   prediction_1  confidence_1  prediction_2  confidence_2  ...
# 0     positive         0.85      negative         0.15  ...
```

### `explain` - Token Attributions

Get token-level importance scores using Captum:

```python
# Get explanations
result = model.predict(data, params={"explain": True})

# Result includes tokens and attributions columns
print(result["tokens"][0])       # ['[CLS]', 'amazing', 'product', '!', '[SEP]', ...]
print(result["attributions"][0]) # [0.05, 0.45, 0.30, 0.10, 0.05, ...]
```

### Combined Parameters

```python
# Top-2 predictions with explanations
result = model.predict(data, params={"top_k": 2, "explain": True})
```

## How It Works

### Training Flow

1. **Data Loading**: Reads CSV files into numpy arrays
2. **Tokenizer Training**: Trains a WordPiece tokenizer on training texts
3. **Model Configuration**: Sets up embedding dimensions, categorical features
4. **MLflow Logger**: Creates `MLFlowLogger` for automatic metric logging
5. **Training**: Trains the classifier with validation monitoring
6. **Artifact Export**: Saves model, tokenizer, and config files
7. **Model Logging**: Logs pyfunc model with all artifacts to MLflow

### Pyfunc Wrapper

The `TextClassifierWrapper` class:

- **`load_context()`**: Loads PyTorch model, HuggingFace tokenizer, and configs
- **`_parse_input()`**: Converts various input formats to (texts, categories)
- **`predict()`**: Runs inference with optional top_k and explain parameters
- **`_predict_with_explain()`**: Uses Captum for token attributions

## Metrics Logged

The example logs these metrics to MLflow:

| Metric | Description |
|--------|-------------|
| `train_loss_step` | Training loss per batch |
| `train_loss_epoch` | Training loss per epoch |
| `train_accuracy` | Training accuracy per epoch |
| `val_loss` | Validation loss per epoch |
| `val_accuracy` | Validation accuracy per epoch |
| `final_train_accuracy` | Final training accuracy |
| `final_val_accuracy` | Final validation accuracy |
| `test_accuracy` | Test set accuracy |

## Troubleshooting

### FutureWarning about filesystem tracking

```
FutureWarning: Relying on the default value of `tracking_uri` is deprecated
```

This is a non-breaking warning from MLflow. To suppress it, set the tracking URI:

```python
mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Or your preferred backend
```

### Model requires torchTextClassifiers

The model uses `torch.save()` which pickles the model class. This requires
`torchTextClassifiers` to be installed when loading. The pip requirements
are included in the logged model metadata.

### Attributions are uniform

Ensure you're using the full PyTorch model (not TorchScript). TorchScript models
don't propagate gradients properly for Captum attribution methods.

## Customization

To adapt this example for your own data:

1. Replace the CSV files in `data/` with your own data
2. Update `category_mapping` in `run_example.py` for your categories
3. Update `label_mapping` for your class labels
4. Adjust model hyperparameters as needed
