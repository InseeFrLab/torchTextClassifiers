# MLflow Integration

Learn how to track experiments, log training metrics, and deploy models with MLflow.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Log training metrics per epoch using PyTorch Lightning's MLFlowLogger
- Create a pyfunc wrapper for flexible model deployment
- Export models with full Captum explainability support
- Handle flexible input formats in production environments
- Use inference parameters (`top_k`, `explain`) for advanced predictions
- Use the MLflow UI to visualize training progress

## Prerequisites

Before starting this tutorial, you should:

- Complete the {doc}`mixed_features` tutorial
- Have MLflow installed (`pip install mlflow` or `uv add mlflow`)
- Have Captum installed for explainability (`pip install captum`)
- Understand PyTorch Lightning basics
- Be familiar with model deployment concepts

## Overview

MLflow is an open-source platform for managing the machine learning lifecycle. In this tutorial, we'll integrate torchTextClassifiers with MLflow to:

1. **Track experiments**: Log parameters, metrics, and artifacts for reproducibility
2. **Monitor training**: Visualize loss and accuracy curves in real-time
3. **Deploy models**: Create portable models with explainability support

### Why Full PyTorch Models?

Our approach saves the full PyTorch model (not TorchScript) to enable:

- **Captum explainability**: Token-level attributions via Integrated Gradients
- **Full model access**: Access to embedding layers for gradient computation
- **Flexible deployment**: Works with any deployment environment

The trade-off is that `torchTextClassifiers` must be installed in the inference environment. The pip requirements are automatically included in the logged model.

## Running the Example

A complete working example is provided in the `examples/mlflow_logging/` directory:

```bash
# From the repository root
uv run examples/mlflow_logging/run_example.py

# Or with explicit dependencies
uv run --extra huggingface --with mlflow --with captum \
    examples/mlflow_logging/run_example.py
```

### Example Structure

```
examples/mlflow_logging/
├── __init__.py           # Package exports
├── pyfunc_wrapper.py     # TextClassifierWrapper class
├── run_example.py        # Main training script
├── README.md             # Detailed documentation
└── data/
    ├── train.csv         # Training data (30 samples)
    ├── val.csv           # Validation data (6 samples)
    └── test.csv          # Test data (6 samples)
```

## Complete Code

Here's the core workflow. See `examples/mlflow_logging/run_example.py` for the full implementation.

```python
import json
import os
import tempfile

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, ParamSchema, ParamSpec
from pytorch_lightning.loggers import MLFlowLogger

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Import the pyfunc wrapper
from examples.mlflow_logging import TextClassifierWrapper


def main():
    # Step 1: Load data from CSV files
    data_dir = Path("examples/mlflow_logging/data")
    train_df = pd.read_csv(data_dir / "train.csv")
    X_train = np.array([[row["text"], row["category"]] for _, row in train_df.iterrows()], dtype=object)
    y_train = train_df["label"].values

    # Step 2: Train tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=1000, output_dim=64)
    tokenizer.train(X_train[:, 0].tolist())

    # Step 3: Configure model
    model_config = ModelConfig(
        embedding_dim=32,
        num_classes=2,
        categorical_vocabulary_sizes=[3],
        categorical_embedding_dims=8,
    )
    classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

    # Step 4: Create MLFlowLogger
    mlflow_logger = MLFlowLogger(
        experiment_name="text-classification",
        log_model=False,
    )

    # Step 5: Train with metric logging
    training_config = TrainingConfig(
        num_epochs=15, batch_size=8, lr=1e-3,
        trainer_params={"logger": mlflow_logger},
    )
    classifier.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)

    # Step 6: Log artifacts to MLflow
    run_id = mlflow_logger.run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"embedding_dim": 32, "num_classes": 2, "vocab_size": 1000})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save tokenizer (HuggingFace format)
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            classifier.tokenizer.tokenizer.save_pretrained(tokenizer_path)

            # Save PyTorch model (full model for Captum support)
            model_path = os.path.join(tmpdir, "model.pt")
            pytorch_model = classifier.pytorch_model
            pytorch_model.eval()
            torch.save(pytorch_model, model_path)

            # Save configs
            with open(os.path.join(tmpdir, "label_mapping.json"), "w") as f:
                json.dump({"0": "negative", "1": "positive"}, f)
            with open(os.path.join(tmpdir, "model_config.json"), "w") as f:
                json.dump({"output_dim": 64, "num_classes": 2, "categorical_columns": ["category"]}, f)

            # Define params schema for inference parameters
            params_schema = ParamSchema([
                ParamSpec("top_k", DataType.long, default=1),
                ParamSpec("explain", DataType.boolean, default=False),
            ])
            signature = ModelSignature(inputs=None, outputs=None, params=params_schema)

            # Log pyfunc model
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=TextClassifierWrapper(),
                artifacts={
                    "model": model_path,
                    "tokenizer": tokenizer_path,
                    "label_mapping": os.path.join(tmpdir, "label_mapping.json"),
                    "model_config": os.path.join(tmpdir, "model_config.json"),
                },
                pip_requirements=["torch>=2.0", "transformers>=4.30", "pandas", "numpy", "captum", "torchTextClassifiers"],
                signature=signature,
            )

    # Step 7: Test inference
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    print(model.predict([["Great product!", 0]]))

if __name__ == "__main__":
    main()
```

## Step-by-Step Walkthrough

### Step 1: Understanding the PyFunc Wrapper

The `TextClassifierWrapper` class is the heart of our deployment strategy. It inherits from `mlflow.pyfunc.PythonModel` and implements two required methods:

```python
class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Called once when the model is loaded."""
        import torch
        from transformers import AutoTokenizer

        # Load full PyTorch model (enables Captum explainability)
        self.model = torch.load(context.artifacts["model"], weights_only=False)
        self.model.eval()

        # Load HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer"])

        # Load configurations
        with open(context.artifacts["label_mapping"]) as f:
            self.label_mapping = json.load(f)

    def predict(self, context, model_input, params=None):
        """Called for each prediction request."""
        # Process input, run inference, return results
        ...
```

**Key Points:**
- `load_context()` is called **once** when the model is loaded
- `predict()` is called for **each inference request**
- The `context.artifacts` dictionary maps artifact names to file paths
- The `params` argument receives inference parameters like `top_k` and `explain`

:::{note}
The wrapper imports `torch` and `transformers` inside the methods. This ensures the dependencies are only loaded when the model is used, not when it's defined.
:::

### Step 2: Flexible Input Handling

The `_parse_input()` method accepts multiple input formats for production flexibility:

```python
def _parse_input(self, model_input):
    # Single string
    if isinstance(model_input, str):
        return [model_input], [[0] * num_cat_features]

    # DataFrame with named columns
    if isinstance(model_input, pd.DataFrame):
        if "text" in model_input.columns:
            texts = model_input["text"].tolist()
            # ... extract categories

    # List of lists: [["text1", cat1], ["text2", cat2]]
    if isinstance(model_input, list):
        if isinstance(model_input[0], list):
            texts = [row[0] for row in model_input]
            categories = [row[1:] for row in model_input]
```

**Supported Formats:**

| Format | Example | Use Case |
|--------|---------|----------|
| Single string | `"Great product!"` | Quick single prediction |
| List of strings | `["Text 1", "Text 2"]` | Batch without categories |
| List of lists | `[["Text", 0], ["Text", 1]]` | Batch with categories |
| DataFrame | `pd.DataFrame({"text": [...], "category": [...]})` | Production pipelines |

### Step 3: Setting Up MLFlowLogger

PyTorch Lightning's `MLFlowLogger` automatically logs training metrics:

```python
from pytorch_lightning.loggers import MLFlowLogger

mlflow_logger = MLFlowLogger(
    experiment_name="text-classification",  # Groups related runs
    log_model=False,  # We'll log manually with pyfunc
)

training_config = TrainingConfig(
    num_epochs=15,
    batch_size=8,
    lr=1e-3,
    trainer_params={"logger": mlflow_logger},  # Pass to Lightning trainer
)
```

**Metrics Logged Automatically:**
- `train_loss_step` - Loss at each training step
- `train_loss_epoch` - Average loss per epoch
- `train_accuracy` - Training accuracy per epoch
- `val_loss` - Validation loss per epoch
- `val_accuracy` - Validation accuracy per epoch

:::{tip}
The `experiment_name` parameter groups related runs together. Use descriptive names like `"sentiment-analysis-v2"` or `"product-reviews"`.
:::

### Step 4: Saving the Model for Captum Support

We save the full PyTorch model (not TorchScript) to enable Captum explainability:

```python
# Get the PyTorch model
pytorch_model = classifier.pytorch_model
pytorch_model.eval()

# Save the full model (enables Captum attribution methods)
torch.save(pytorch_model, "model.pt")
```

**Why Full PyTorch Instead of TorchScript?**
- TorchScript models don't propagate gradients properly
- Captum's `LayerIntegratedGradients` requires gradient access
- The full model preserves access to embedding layers

:::{note}
The trade-off is that `torchTextClassifiers` must be installed at inference time, since `torch.save()` pickles the model class which includes references to the original module.
:::

### Step 5: Defining the Params Schema

To enable inference parameters (`top_k`, `explain`), define a params schema:

```python
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, ParamSchema, ParamSpec

params_schema = ParamSchema([
    ParamSpec("top_k", DataType.long, default=1),
    ParamSpec("explain", DataType.boolean, default=False),
])

signature = ModelSignature(
    inputs=None,   # Flexible input formats
    outputs=None,  # Output varies based on params
    params=params_schema,
)
```

:::{important}
Without a params schema, MLflow ignores the `params` argument in `predict()`. The schema explicitly declares which parameters are accepted.
:::

### Step 6: Logging the PyFunc Model

Finally, we log everything to MLflow:

```python
with mlflow.start_run(run_id=mlflow_logger.run_id):
    # Log hyperparameters
    mlflow.log_params({
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "vocab_size": vocab_size,
    })

    # Log the pyfunc model with all artifacts
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TextClassifierWrapper(),
        artifacts={
            "model": model_path,
            "tokenizer": tokenizer_path,
            "label_mapping": label_mapping_path,
            "model_config": model_config_path,
        },
        pip_requirements=[
            "torch>=2.0",
            "transformers>=4.30",
            "pandas",
            "numpy",
            "captum",
            "torchTextClassifiers",
        ],
        signature=signature,
    )
```

**Artifacts Saved:**

| Artifact | Format | Purpose |
|----------|--------|---------|
| `model.pt` | PyTorch | The full PyTorch model (for Captum) |
| `tokenizer/` | HuggingFace | Vocabulary and tokenizer config |
| `label_mapping.json` | JSON | `{"0": "negative", "1": "positive"}` |
| `model_config.json` | JSON | Model configuration |

## Using the MLflow UI

After training, launch the MLflow UI to visualize your experiments:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Viewing Training Curves

1. Select your experiment ("text-classification")
2. Click on a run
3. Go to the "Metrics" tab
4. View `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` over epochs

### Comparing Runs

1. Select multiple runs using checkboxes
2. Click "Compare"
3. View side-by-side metrics and parameters
4. Identify the best-performing configuration

### Accessing Artifacts

1. Click on a run
2. Go to the "Artifacts" tab
3. Browse the saved files:
   - `model/` - Contains the pyfunc model
   - `model/artifacts/` - PyTorch model, tokenizer, configs

## Loading and Using the Model

Once logged, load the model anywhere MLflow is installed:

```python
import mlflow

# Load the model
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Make predictions with flexible input formats
model.predict("Great product!")
model.predict(["Text 1", "Text 2"])
model.predict([["Text with category", 0]])
model.predict(pd.DataFrame({"text": ["Hello"], "category": [1]}))
```

:::{note}
Replace `<run_id>` with the actual run ID from your training session. You can find it in the MLflow UI or in the training output.
:::

## Inference Parameters

The pyfunc wrapper supports additional inference parameters via the `params` argument:

### top_k: Multiple Predictions

Return the top k predictions instead of just the best one:

```python
# Get top 3 predictions per sample
result = model.predict(
    [["This product might be good or bad.", 0]],
    params={"top_k": 3}
)
print(result)
#   prediction_1  confidence_1  prediction_2  confidence_2
# 0     positive          0.52      negative          0.48
```

**Output Columns with top_k:**
- `prediction_1`, `confidence_1` - Best prediction
- `prediction_2`, `confidence_2` - Second best
- ... up to `prediction_k`, `confidence_k`

:::{tip}
If `top_k` exceeds the number of classes, it's automatically limited to the number of classes.
:::

### explain: Token Attributions

Get token-level attributions to understand which parts of the input influenced the prediction:

```python
# Get token attributions
result = model.predict(
    [["Amazing quality product!"]],
    params={"explain": True}
)
print(result)
#   prediction  confidence                    tokens                attributions
# 0   positive       0.92  [amazing, quality, prod...]  [0.35, 0.28, 0.15, ...]

# Analyze which tokens were most important
tokens = result.iloc[0]["tokens"]
attributions = result.iloc[0]["attributions"]
for tok, attr in sorted(zip(tokens, attributions), key=lambda x: x[1], reverse=True)[:5]:
    if tok != "[PAD]":
        print(f"  {tok}: {attr:.4f}")
# Output:
#   amazing: 0.3500
#   quality: 0.2800
#   product: 0.1500
#   !: 0.0200
```

**Output Columns with explain:**
- `prediction`, `confidence` - The prediction result
- `tokens` - List of tokenized words
- `attributions` - Normalized importance scores (sum to 1.0)

:::{note}
The attributions use Captum's `LayerIntegratedGradients`, which computes gradient-based feature importance through the embedding layer.
:::

### Combined Parameters

Use multiple parameters together:

```python
# Get top 2 predictions with explanations
result = model.predict(
    [["Great product but shipping was slow."]],
    params={"top_k": 2, "explain": True}
)
print(result.columns.tolist())
# ['prediction_1', 'confidence_1', 'prediction_2', 'confidence_2', 'tokens', 'attributions']
```

### Summary of Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 1 | Number of top predictions to return |
| `explain` | bool | False | Include token attributions |

**Example:**
```python
# Default behavior (top_k=1, explain=False)
model.predict(data)

# Get top 3 predictions
model.predict(data, params={"top_k": 3})

# Get attributions
model.predict(data, params={"explain": True})

# Get both
model.predict(data, params={"top_k": 2, "explain": True})
```

## Common Issues and Solutions

### Issue: "Run not found" Error

**Symptoms:** `MlflowException: Run 'xxx' not found`

**Solutions:**
1. Ensure you're using the same tracking URI:
   ```python
   mlflow.set_tracking_uri("file:./mlruns")  # Local
   mlflow.set_tracking_uri("http://localhost:5000")  # Remote server
   ```
2. Check the `mlruns/` directory exists in your working directory
3. Verify the run ID is correct

### Issue: Uniform Attributions

**Symptoms:** All attribution values are the same (e.g., 0.0156)

**Solutions:**
1. Ensure you're using the full PyTorch model, not TorchScript
2. The model must be loaded with `torch.load()`, not `torch.jit.load()`
3. Verify Captum has access to the embedding layer

### Issue: Missing Dependencies at Inference

**Symptoms:** `ModuleNotFoundError` when loading the model

**Solutions:**
1. Install the required packages:
   ```bash
   pip install torch transformers pandas numpy captum torchTextClassifiers
   ```
2. Or use the generated `requirements.txt`:
   ```bash
   pip install -r mlruns/<experiment_id>/<run_id>/artifacts/model/requirements.txt
   ```

### Issue: Input Format Errors

**Symptoms:** `ValueError: Unsupported input format`

**Solutions:**
1. Check your input is one of the supported formats
2. For DataFrames, ensure the "text" column exists
3. For lists of lists, ensure each inner list has `[text, category]` format

### Issue: Params Ignored

**Symptoms:** `params={"top_k": 3}` has no effect

**Solutions:**
1. Ensure the model was logged with a `ModelSignature` that includes a `ParamSchema`
2. The params schema must be defined before logging:
   ```python
   params_schema = ParamSchema([
       ParamSpec("top_k", DataType.long, default=1),
       ParamSpec("explain", DataType.boolean, default=False),
   ])
   signature = ModelSignature(inputs=None, outputs=None, params=params_schema)
   ```

## Production Deployment

### Serving with MLflow

Start a REST API server:

```bash
mlflow models serve -m runs:/<run_id>/model -p 5001
```

Make predictions via HTTP:

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [["Great product!", 0]]}'
```

### Loading in a Fresh Environment

```python
# In a new environment with the required dependencies
import mlflow

# Load from MLflow tracking server
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
model = mlflow.pyfunc.load_model("models:/text-classifier/Production")

# Or load from a local path
model = mlflow.pyfunc.load_model("/path/to/model")

# Make predictions
result = model.predict(["This is great!"])
print(result)
#   prediction  confidence
# 0   positive       0.89
```

## Next Steps

- **MLflow Model Registry**: Learn to version and stage models with [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- **Remote Tracking**: Set up a [remote MLflow tracking server](https://mlflow.org/docs/latest/tracking.html#tracking-server)
- **Custom Metrics**: Add custom metrics by modifying the Lightning module
- **Batch Inference**: Process large datasets efficiently with batch predictions

## Summary

In this tutorial, you learned:

- How to integrate PyTorch Lightning with MLflow for automatic metric logging
- How to create a pyfunc wrapper for flexible model deployment
- How to save models with full Captum explainability support
- How to handle flexible input formats in production
- How to use inference parameters (`top_k` for multiple predictions, `explain` for attributions)
- How to use the MLflow UI to visualize training progress
- How to serve models via REST API

Your models are now ready for production deployment with full experiment tracking, explainability, and reproducibility!
