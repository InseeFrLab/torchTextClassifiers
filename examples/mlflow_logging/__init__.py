"""
MLflow Integration Example
==========================

This package demonstrates how to train a text classifier with torchTextClassifiers
and log it to MLflow with a pyfunc wrapper for flexible deployment.

Features:
    - Train a text classifier with text + categorical features
    - Log training metrics per epoch via PyTorch Lightning's MLFlowLogger
    - Create a pyfunc wrapper for inference with multiple input formats
    - Use Captum for model explainability (token attributions)

Package Structure:
    - pyfunc_wrapper.py: TextClassifierWrapper class for MLflow pyfunc
    - run_example.py: Main script demonstrating the complete workflow
    - data/: CSV files with sample training, validation, and test data

Usage:
    Run the example script:

    .. code-block:: bash

        uv run examples/mlflow_logging/run_example.py

    Or import the wrapper for custom use:

    .. code-block:: python

        from examples.mlflow_logging import TextClassifierWrapper

Example:
    After running the example, load the model:

    .. code-block:: python

        import mlflow
        model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

        # Basic prediction
        model.predict(["Great product!"])

        # With parameters
        model.predict(data, params={"top_k": 3, "explain": True})
"""

from .pyfunc_wrapper import TextClassifierWrapper

__all__ = ["TextClassifierWrapper"]
