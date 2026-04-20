#!/usr/bin/env python
"""
MLflow Logging Example - Main Script
=====================================

This script demonstrates the complete workflow for training a text classifier
with torchTextClassifiers and logging it to MLflow with a pyfunc wrapper.

The workflow consists of:
1. Loading training data from CSV files
2. Training a WordPiece tokenizer
3. Configuring and training the model with MLflow logging
4. Exporting artifacts (model, tokenizer, configs)
5. Logging a pyfunc model to MLflow
6. Testing inference with various input formats

Usage:
    uv run examples/mlflow_logging/run_example.py

    # Or with explicit dependencies:
    uv run --extra huggingface --with mlflow --with captum \\
        examples/mlflow_logging/run_example.py

Requirements:
    - torchTextClassifiers[huggingface]
    - mlflow
    - captum (for explainability)

Output:
    - MLflow run with training metrics logged per epoch
    - Logged pyfunc model ready for deployment
    - Test predictions demonstrating various input formats
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

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

# Import the pyfunc wrapper from the same package
# Handle both direct script execution and module import
try:
    from .pyfunc_wrapper import TextClassifierWrapper
except ImportError:
    from pyfunc_wrapper import TextClassifierWrapper

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


# =============================================================================
# Configuration (from environment variables or defaults)
# =============================================================================
# These can be overridden via environment variables for Argo Workflows
# or other orchestration systems.


def get_config() -> dict:
    """
    Get training configuration from environment variables or defaults.

    Environment Variables:
        NUM_WORKERS: Number of data loader workers (default: 0)
        NUM_EPOCHS: Number of training epochs (default: 15)
        BATCH_SIZE: Training batch size (default: 8)
        LEARNING_RATE: Learning rate (default: 0.001)
        VOCAB_SIZE: Tokenizer vocabulary size (default: 1000)
        EMBEDDING_DIM: Token embedding dimension (default: 32)
        OUTPUT_DIM: Maximum sequence length (default: 64)
        EXPERIMENT_NAME: MLflow experiment name (default: text-classification)

    Returns:
        Dictionary with configuration values.
    """
    return {
        "num_workers": int(os.environ.get("NUM_WORKERS", 0)),
        "num_epochs": int(os.environ.get("NUM_EPOCHS", 15)),
        "batch_size": int(os.environ.get("BATCH_SIZE", 8)),
        "learning_rate": float(os.environ.get("LEARNING_RATE", 0.001)),
        "vocab_size": int(os.environ.get("VOCAB_SIZE", 1000)),
        "embedding_dim": int(os.environ.get("EMBEDDING_DIM", 32)),
        "output_dim": int(os.environ.get("OUTPUT_DIM", 64)),
        "experiment_name": os.environ.get("EXPERIMENT_NAME", "text-classification"),
    }


# =============================================================================
# Data Loading
# =============================================================================


def load_data(data_dir: Path) -> tuple:
    """
    Load training, validation, and test data from CSV files.

    The CSV files should have columns: text, category, label
    - text: The text content to classify
    - category: Categorical feature (0=electronics, 1=clothing, 2=books)
    - label: Target label (0=negative, 1=positive)

    Args:
        data_dir: Path to the directory containing train.csv, val.csv, test.csv

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) where:
            - X arrays have shape (n_samples, 2) with [text, category]
            - y arrays have shape (n_samples,) with labels

    Example:
        >>> data_dir = Path("examples/mlflow_logging/data")
        >>> X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)
        >>> print(f"Training samples: {len(X_train)}")
    """
    def load_csv(filename: str) -> tuple:
        """Load a single CSV file and convert to arrays."""
        df = pd.read_csv(data_dir / filename)
        # X: [text, category] as object array
        X = np.array(
            [[row["text"], row["category"]] for _, row in df.iterrows()],
            dtype=object
        )
        # y: labels as integer array
        y = df["label"].values
        return X, y

    X_train, y_train = load_csv("train.csv")
    X_val, y_val = load_csv("val.csv")
    X_test, y_test = load_csv("test.csv")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# Main Training and Logging Function
# =============================================================================


def main():
    """
    Main function demonstrating the complete MLflow logging workflow.

    This function:
    1. Loads data from CSV files
    2. Trains a tokenizer and classifier
    3. Logs training metrics to MLflow via PyTorch Lightning
    4. Exports and logs a pyfunc model
    5. Tests the loaded model with various input formats
    """
    print("=" * 60)
    print("MLflow Logging Example")
    print("=" * 60)

    # =========================================================================
    # Load Configuration
    # =========================================================================
    config = get_config()
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # =========================================================================
    # Step 1: Load Data from CSV Files
    # =========================================================================
    print("\n1. Loading data from CSV files...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")

    # Extract text for tokenizer training
    training_texts = X_train[:, 0].tolist()

    # =========================================================================
    # Step 2: Configure and Train Tokenizer
    # =========================================================================
    print("\n2. Training tokenizer...")

    # Tokenizer hyperparameters (from config)
    vocab_size = config["vocab_size"]
    output_dim = config["output_dim"]

    tokenizer = WordPieceTokenizer(vocab_size=vocab_size, output_dim=output_dim)
    tokenizer.train(training_texts)

    print(f"   Vocabulary size: {len(tokenizer)}")

    # =========================================================================
    # Step 3: Configure Model
    # =========================================================================
    print("\n3. Creating classifier...")

    # Model hyperparameters (from config)
    embedding_dim = config["embedding_dim"]
    num_classes = 2                 # Binary classification (positive/negative)
    categorical_vocab_sizes = [3]   # 3 categories: electronics, clothing, books
    categorical_embedding_dims = 8  # Dimension for categorical embeddings

    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        categorical_vocabulary_sizes=categorical_vocab_sizes,
        categorical_embedding_dims=categorical_embedding_dims,
    )

    # Create the classifier
    classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

    # Label and category mappings for inference
    label_mapping = {"0": "negative", "1": "positive"}
    category_mapping = {"category": {"electronics": 0, "clothing": 1, "books": 2}}

    # =========================================================================
    # Step 4: Configure MLflow Logger
    # =========================================================================
    # PyTorch Lightning's MLFlowLogger automatically logs metrics per epoch:
    # - train_loss_step, train_loss_epoch
    # - train_accuracy
    # - val_loss, val_accuracy

    print("\n4. Training with MLflow logging...")

    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        log_model=False,  # We'll log manually with pyfunc wrapper
    )

    # Training hyperparameters (from config)
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    num_workers = config["num_workers"]

    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        patience_early_stopping=5,
        num_workers=num_workers,
        trainer_params={"logger": mlflow_logger},
    )

    # =========================================================================
    # Step 5: Train the Model
    # =========================================================================
    # Training metrics are logged automatically per epoch
    classifier.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    # Get the run_id from MLFlowLogger to continue logging in the same run
    run_id = mlflow_logger.run_id

    # =========================================================================
    # Step 6: Log Additional Metrics and Artifacts
    # =========================================================================
    with mlflow.start_run(run_id=run_id):
        # Log hyperparameters
        mlflow.log_params({
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "vocab_size": vocab_size,
            "output_dim": output_dim,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_workers": num_workers,
            "categorical_vocab_sizes": str(categorical_vocab_sizes),
            "categorical_embedding_dims": categorical_embedding_dims,
        })

        # -----------------------------------------------------------------
        # Evaluate on all sets
        # -----------------------------------------------------------------
        print("\n5. Evaluating model...")

        def evaluate(X, y, name):
            result = classifier.predict(X)
            preds = result["prediction"].squeeze().numpy()
            acc = (preds == y).mean()
            print(f"   {name} accuracy: {acc:.4f}")
            return acc

        train_acc = evaluate(X_train, y_train, "Training")
        val_acc = evaluate(X_val, y_val, "Validation")
        test_acc = evaluate(X_test, y_test, "Test")

        # Log final evaluation metrics
        mlflow.log_metrics({
            "final_train_accuracy": train_acc,
            "final_val_accuracy": val_acc,
            "test_accuracy": test_acc,
        })

        # -----------------------------------------------------------------
        # Prepare artifacts for logging
        # -----------------------------------------------------------------
        print("\n6. Preparing artifacts for logging...")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save tokenizer (HuggingFace format)
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            classifier.tokenizer.tokenizer.save_pretrained(tokenizer_path)
            print(f"   Saved tokenizer to {tokenizer_path}")

            # Save PyTorch model (full model for Captum support)
            model_path = os.path.join(tmpdir, "model.pt")
            pytorch_model = classifier.pytorch_model
            pytorch_model.eval()
            torch.save(pytorch_model, model_path)
            print(f"   Saved PyTorch model to {model_path}")

            # Save label mapping
            label_mapping_path = os.path.join(tmpdir, "label_mapping.json")
            with open(label_mapping_path, "w") as f:
                json.dump(label_mapping, f)
            print(f"   Saved label mapping to {label_mapping_path}")

            # Save model config
            model_config_path = os.path.join(tmpdir, "model_config.json")
            config_dict = {
                "output_dim": output_dim,
                "num_classes": num_classes,
                "embedding_dim": embedding_dim,
                "categorical_columns": ["category"],
            }
            with open(model_config_path, "w") as f:
                json.dump(config_dict, f)
            print(f"   Saved model config to {model_config_path}")

            # Save categorical mapping
            categorical_mapping_path = os.path.join(tmpdir, "categorical_mapping.json")
            with open(categorical_mapping_path, "w") as f:
                json.dump(category_mapping, f)
            print(f"   Saved categorical mapping to {categorical_mapping_path}")

            # Define artifacts dictionary
            artifacts = {
                "model": model_path,
                "tokenizer": tokenizer_path,
                "label_mapping": label_mapping_path,
                "model_config": model_config_path,
                "categorical_mapping": categorical_mapping_path,
            }

            # Define pip requirements
            pip_requirements = [
                "torch>=2.0",
                "transformers>=4.30",
                "pandas",
                "numpy",
                "captum",
                "torchTextClassifiers",
            ]

            # -----------------------------------------------------------------
            # Define model signature with params schema
            # -----------------------------------------------------------------
            # This enables the params argument (top_k, explain) in predict()
            params_schema = ParamSchema([
                ParamSpec("top_k", DataType.long, default=1),
                ParamSpec("explain", DataType.boolean, default=False),
            ])
            signature = ModelSignature(
                inputs=None,   # Flexible input formats
                outputs=None,  # Output varies based on params
                params=params_schema,
            )

            # -----------------------------------------------------------------
            # Log the pyfunc model
            # -----------------------------------------------------------------
            print("\n7. Logging pyfunc model to MLflow...")

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=TextClassifierWrapper(),
                artifacts=artifacts,
                pip_requirements=pip_requirements,
                signature=signature,
            )

        print(f"\n   Run ID: {run_id}")

        # =====================================================================
        # Step 7: Test Model Loading and Inference
        # =====================================================================
        print("\n8. Testing model loading and inference...")

        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # -----------------------------------------------------------------
        # Test different input formats
        # -----------------------------------------------------------------
        print("\n   Testing different input formats:")

        # Format 1: List of lists
        print("\n   Format 1: List of lists [['text', category], ...]")
        preds = loaded_model.predict([["This is an amazing product!", 0]])
        print(f"   -> {preds.iloc[0]['prediction']} ({preds.iloc[0]['confidence']:.3f})")

        # Format 2: Multiple samples
        print("\n   Format 2: Multiple samples")
        preds = loaded_model.predict([
            ["Great quality electronics!", 0],
            ["Terrible fit, returning it.", 1],
        ])
        for i, p in preds.iterrows():
            print(f"   -> {p['prediction']} ({p['confidence']:.3f})")

        # Format 3: List of strings (text only)
        print("\n   Format 3: List of strings (text only)")
        preds = loaded_model.predict(["Love this product!", "Worst purchase ever."])
        for i, p in preds.iterrows():
            print(f"   -> {p['prediction']} ({p['confidence']:.3f})")

        # Format 4: Single string
        print("\n   Format 4: Single string")
        preds = loaded_model.predict("Absolutely fantastic!")
        print(f"   -> {preds.iloc[0]['prediction']} ({preds.iloc[0]['confidence']:.3f})")

        # Format 5: DataFrame
        print("\n   Format 5: DataFrame with named columns")
        test_df = pd.DataFrame({"text": ["Beautiful design!"], "category": [1]})
        preds = loaded_model.predict(test_df)
        print(f"   -> {preds.iloc[0]['prediction']} ({preds.iloc[0]['confidence']:.3f})")

        # -----------------------------------------------------------------
        # Test inference parameters
        # -----------------------------------------------------------------
        print("\n   Testing inference parameters:")

        # Test top_k parameter
        print("\n   Parameter: top_k=2 (get top 2 predictions)")
        preds = loaded_model.predict(
            [["This could be good or bad, not sure.", 0]],
            params={"top_k": 2},
        )
        print(f"   -> Top 1: {preds.iloc[0]['prediction_1']} ({preds.iloc[0]['confidence_1']:.3f})")
        print(f"   -> Top 2: {preds.iloc[0]['prediction_2']} ({preds.iloc[0]['confidence_2']:.3f})")

        # Test explain parameter
        print("\n   Parameter: explain=True (get token attributions)")
        preds = loaded_model.predict(
            [["Amazing quality product!", 0]],
            params={"explain": True},
        )
        print(f"   -> Prediction: {preds.iloc[0]['prediction']} ({preds.iloc[0]['confidence']:.3f})")
        tokens = preds.iloc[0]["tokens"]
        attributions = preds.iloc[0]["attributions"]
        # Show top 5 most important tokens (excluding padding)
        token_attr_pairs = [
            (t, a) for t, a in zip(tokens, attributions) if t != "[PAD]"
        ]
        token_attr_pairs.sort(key=lambda x: x[1], reverse=True)
        print("   -> Top tokens by attribution:")
        for tok, attr in token_attr_pairs[:5]:
            print(f"      '{tok}': {attr:.4f}")

        # Test combined top_k and explain
        print("\n   Parameters: top_k=2, explain=True (combined)")
        preds = loaded_model.predict(
            [["Great product but shipping was slow.", 0]],
            params={"top_k": 2, "explain": True},
        )
        print(f"   -> Top 1: {preds.iloc[0]['prediction_1']} ({preds.iloc[0]['confidence_1']:.3f})")
        print(f"   -> Top 2: {preds.iloc[0]['prediction_2']} ({preds.iloc[0]['confidence_2']:.3f})")
        print(f"   -> Tokens: {preds.iloc[0]['tokens'][:10]}...")

    # =========================================================================
    # Print Usage Instructions
    # =========================================================================
    print("\n" + "=" * 60)
    print("Done! Model logged successfully to MLflow.")
    print(f"\nTo load and use the model:")
    print()
    print(f"  import mlflow")
    print(f"  model = mlflow.pyfunc.load_model('runs:/{run_id}/model')")
    print()
    print(f"  # Flexible input formats supported:")
    print(f'  model.predict([["Great product!", 0]])           # List of [text, category]')
    print(f'  model.predict(["Text 1", "Text 2"])              # List of strings')
    print(f'  model.predict("Single text")                     # Single string')
    print(f'  model.predict(pd.DataFrame(...))                 # DataFrame')
    print()
    print(f"  # Inference parameters (via params dict):")
    print(f'  model.predict(data, params={{"top_k": 3}})        # Get top 3 predictions')
    print(f'  model.predict(data, params={{"explain": True}})   # Get token attributions')
    print(f'  model.predict(data, params={{"top_k": 2, "explain": True}})  # Combined')
    print("=" * 60)


if __name__ == "__main__":
    main()
