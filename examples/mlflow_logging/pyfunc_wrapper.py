"""
PyFunc Wrapper for Text Classification Models
==============================================

This module provides an MLflow pyfunc wrapper for text classification models
trained with torchTextClassifiers. The wrapper enables:

- **Flexible input formats**: Accept strings, lists, DataFrames
- **Multiple predictions**: Return top-k predictions per sample
- **Explainability**: Token-level attributions via Captum

The wrapper is designed to work with models logged to MLflow and can be
loaded in any environment with the required dependencies.

Dependencies:
    - torch>=2.0
    - transformers>=4.30
    - pandas
    - numpy
    - captum (for explainability)
    - torchTextClassifiers (for model class)

Example:
    >>> import mlflow
    >>> model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
    >>> model.predict(["Great product!"])
    >>> model.predict(data, params={"top_k": 3, "explain": True})
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow.pyfunc
import numpy as np
import pandas as pd


class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for text classification.

    This wrapper loads a PyTorch model and HuggingFace tokenizer,
    enabling inference with full explainability support via Captum.

    Attributes:
        model: The loaded PyTorch text classification model.
        tokenizer: HuggingFace tokenizer for text preprocessing.
        label_mapping: Dict mapping class indices to label names.
        config: Model configuration (output_dim, num_classes, etc.).
        categorical_mapping: Optional mapping for categorical features.

    Supported Input Formats:
        - Single string: "text"
        - List of strings: ["text1", "text2"]
        - List of lists: [["text1", cat1], ["text2", cat2]]
        - DataFrame with columns: pd.DataFrame({"text": [...], "category": [...]})
        - DataFrame positional: pd.DataFrame([["text1", 0], ["text2", 1]])

    Inference Parameters (via params dict):
        - top_k (int): Number of top predictions to return (default: 1)
        - explain (bool): Return token attributions (default: False)

    Example:
        >>> # Basic prediction
        >>> model.predict(["Great product!"])

        >>> # Top-3 predictions
        >>> model.predict(data, params={"top_k": 3})

        >>> # With explainability
        >>> model.predict(data, params={"explain": True})
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load model artifacts from MLflow context.

        This method is called once when the model is loaded. It initializes:
        - The PyTorch model from the saved checkpoint
        - The HuggingFace tokenizer
        - Label mapping and model configuration

        Args:
            context: MLflow context containing artifact paths.
        """
        import json

        import torch
        from transformers import AutoTokenizer

        # =====================================================================
        # Load PyTorch Model
        # =====================================================================
        # The model is saved as a full PyTorch model (not TorchScript)
        # to enable Captum-based explainability
        model_path = context.artifacts["model"]
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()

        # =====================================================================
        # Load Tokenizer
        # =====================================================================
        # HuggingFace tokenizer saved with save_pretrained()
        tokenizer_path = context.artifacts["tokenizer"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # =====================================================================
        # Load Configurations
        # =====================================================================
        # Label mapping: {"0": "negative", "1": "positive"}
        with open(context.artifacts["label_mapping"]) as f:
            self.label_mapping = json.load(f)

        # Model config: output_dim, num_classes, categorical_columns
        with open(context.artifacts["model_config"]) as f:
            self.config = json.load(f)

        # Optional: categorical feature mapping
        if "categorical_mapping" in context.artifacts:
            with open(context.artifacts["categorical_mapping"]) as f:
                self.categorical_mapping = json.load(f)
        else:
            self.categorical_mapping = {}

    def _parse_input(
        self, model_input: Any
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Parse various input formats into (texts, categories).

        This method handles multiple input formats for flexibility in
        production environments where data may come in different shapes.

        Args:
            model_input: Input data in one of the supported formats.

        Returns:
            Tuple of (texts, categories) where:
                - texts: List of text strings
                - categories: List of category values per sample

        Raises:
            ValueError: If input format is not supported.

        Examples:
            >>> # Single string
            >>> texts, cats = self._parse_input("Hello world")
            >>> # texts = ["Hello world"], cats = [[0]]

            >>> # List of lists with categories
            >>> texts, cats = self._parse_input([["Text", 1], ["Other", 2]])
            >>> # texts = ["Text", "Other"], cats = [[1], [2]]
        """
        num_cat_features = len(self.config.get("categorical_columns", []))

        # -----------------------------------------------------------------
        # Case 1: Single string
        # -----------------------------------------------------------------
        if isinstance(model_input, str):
            texts = [model_input]
            categories = [[0] * num_cat_features] if num_cat_features else []
            return texts, categories

        # -----------------------------------------------------------------
        # Case 2: DataFrame
        # -----------------------------------------------------------------
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                # Named columns: DataFrame({"text": [...], "category": [...]})
                texts = model_input["text"].tolist()
                cat_cols = self.config.get("categorical_columns", [])
                if cat_cols and all(c in model_input.columns for c in cat_cols):
                    categories = model_input[cat_cols].values.tolist()
                else:
                    categories = [[0] * num_cat_features] * len(texts)
            else:
                # Positional: first column = text, rest = categories
                texts = model_input.iloc[:, 0].tolist()
                if model_input.shape[1] > 1:
                    categories = model_input.iloc[:, 1:].values.tolist()
                else:
                    categories = [[0] * num_cat_features] * len(texts)
            return texts, categories

        # -----------------------------------------------------------------
        # Case 3: List or Array
        # -----------------------------------------------------------------
        if isinstance(model_input, (list, np.ndarray)):
            model_input = list(model_input)

            if len(model_input) == 0:
                return [], []

            # Check if first element is a list/array (batch of samples)
            if isinstance(model_input[0], (list, np.ndarray)):
                # List of lists: [["text1", cat1], ["text2", cat2]]
                texts = [row[0] for row in model_input]
                if len(model_input[0]) > 1:
                    categories = [list(row[1:]) for row in model_input]
                else:
                    categories = [[0] * num_cat_features] * len(texts)
                return texts, categories

            # First element is not a list
            if all(isinstance(x, str) for x in model_input):
                # List of strings: ["text1", "text2"]
                return model_input, [[0] * num_cat_features] * len(model_input)
            else:
                # Single sample: ["text", cat1, cat2]
                texts = [str(model_input[0])]
                if len(model_input) > 1:
                    categories = [list(model_input[1:])]
                else:
                    categories = [[0] * num_cat_features]
                return texts, categories

        raise ValueError(f"Unsupported input format: {type(model_input)}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Run inference on input data.

        This method handles both standard predictions and explainability.
        When explain=True, it uses Captum's LayerIntegratedGradients to
        compute token-level attributions.

        Args:
            context: MLflow context (not used directly).
            model_input: Input data in any supported format.
            params: Optional inference parameters:
                - top_k (int): Number of top predictions (default: 1)
                - explain (bool): Return attributions (default: False)

        Returns:
            DataFrame with predictions. Columns depend on params:
            - top_k=1: "prediction", "confidence"
            - top_k>1: "prediction_1", "confidence_1", ..., "prediction_k", "confidence_k"
            - explain=True: adds "tokens", "attributions" columns

        Example:
            >>> # Standard prediction
            >>> result = model.predict(["Great product!"])
            >>> print(result)
            #   prediction  confidence
            # 0   positive       0.89

            >>> # Top-3 with explanations
            >>> result = model.predict(data, params={"top_k": 3, "explain": True})
        """
        import torch

        # =====================================================================
        # Parse Parameters
        # =====================================================================
        params = params or {}
        top_k = params.get("top_k", 1)
        explain = params.get("explain", False)

        # =====================================================================
        # Parse Input
        # =====================================================================
        texts, categories = self._parse_input(model_input)

        if len(texts) == 0:
            return pd.DataFrame({"prediction": [], "confidence": []})

        # =====================================================================
        # Tokenize Text
        # =====================================================================
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config["output_dim"],
            return_tensors="pt",
        )

        # =====================================================================
        # Prepare Categorical Features
        # =====================================================================
        num_cat_features = len(self.config.get("categorical_columns", []))
        if num_cat_features > 0 and categories:
            categorical_vars = torch.tensor(categories, dtype=torch.long)
        else:
            categorical_vars = torch.zeros(len(texts), num_cat_features, dtype=torch.long)

        # Ensure top_k doesn't exceed number of classes
        num_classes = self.config["num_classes"]
        top_k = min(top_k, num_classes)

        # =====================================================================
        # Run Inference (with or without explainability)
        # =====================================================================
        if explain:
            # Use Captum for token-level attributions
            probs, attributions_list, tokens_list = self._predict_with_explain(
                tokens, categorical_vars, texts
            )
        else:
            # Standard inference without gradients
            with torch.no_grad():
                logits = self.model(
                    tokens["input_ids"], tokens["attention_mask"], categorical_vars
                )
                probs = torch.softmax(logits, dim=-1)

        # =====================================================================
        # Build Result DataFrame
        # =====================================================================
        result_data = {}

        if top_k == 1:
            # Single prediction per sample
            predictions = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            pred_labels = [self.label_mapping[str(p.item())] for p in predictions]
            result_data["prediction"] = pred_labels
            result_data["confidence"] = confidence.detach().numpy()
        else:
            # Top-k predictions per sample
            top_probs, top_indices = probs.topk(k=top_k, dim=-1)
            for k_idx in range(top_k):
                pred_labels = [
                    self.label_mapping[str(idx[k_idx].item())] for idx in top_indices
                ]
                result_data[f"prediction_{k_idx + 1}"] = pred_labels
                result_data[f"confidence_{k_idx + 1}"] = (
                    top_probs[:, k_idx].detach().numpy()
                )

        # Add attribution columns if explain=True
        if explain:
            result_data["tokens"] = tokens_list
            result_data["attributions"] = attributions_list

        return pd.DataFrame(result_data)

    def _predict_with_explain(
        self,
        tokens: Dict[str, "torch.Tensor"],
        categorical_vars: "torch.Tensor",
        texts: List[str],
    ) -> Tuple["torch.Tensor", List[List[float]], List[List[str]]]:
        """
        Run inference with Captum explainability.

        Uses LayerIntegratedGradients to compute token-level attributions
        by analyzing gradients through the embedding layer.

        Args:
            tokens: Tokenized input with input_ids and attention_mask.
            categorical_vars: Categorical feature tensor.
            texts: Original text strings (for token decoding).

        Returns:
            Tuple of (probs, attributions_list, tokens_list):
                - probs: Softmax probabilities
                - attributions_list: Normalized attribution scores per sample
                - tokens_list: Decoded tokens per sample
        """
        import torch
        from captum.attr import LayerIntegratedGradients

        # Initialize Captum with the embedding layer
        lig = LayerIntegratedGradients(
            self.model, self.model.text_embedder.embedding_layer
        )

        # Forward pass to get predictions
        with torch.no_grad():
            logits = self.model(
                tokens["input_ids"], tokens["attention_mask"], categorical_vars
            )
            probs = torch.softmax(logits, dim=-1)

        # Get predictions for attribution targets
        predictions = probs.argmax(dim=-1)

        # Compute attributions using LayerIntegratedGradients
        attributions = lig.attribute(
            (tokens["input_ids"], tokens["attention_mask"], categorical_vars),
            target=predictions,
        )
        # Sum over embedding dimension to get per-token attributions
        attributions = attributions.sum(dim=-1)  # (batch_size, seq_len)

        # Normalize attributions per sample
        attributions_list = []
        tokens_list = []
        for i in range(len(texts)):
            attr = attributions[i].abs()
            if attr.sum() > 0:
                attr = attr / attr.sum()
            attributions_list.append(attr.detach().tolist())

            # Decode tokens for this sample
            sample_tokens = self.tokenizer.convert_ids_to_tokens(
                tokens["input_ids"][i].tolist()
            )
            tokens_list.append(sample_tokens)

        return probs, attributions_list, tokens_list
