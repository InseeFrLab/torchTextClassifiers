"""FastText model components.

This module contains the PyTorch model, Lightning module, and dataset classes
for FastText classification. Consolidates what was previously in pytorch_model.py,
lightning_module.py, and dataset.py.
"""

import logging
from typing import Annotated, Optional

import torch
from torch import nn

from torchTextClassifiers.model.components import (
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    TextEmbedder,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


# ============================================================================
# PyTorch Model

# It takes PyTorch tensors as input (not raw text!),
# and it outputs raw not-softmaxed logits, not predictions
# ============================================================================


class TextClassificationModel(nn.Module):
    """FastText Pytorch Model."""

    def __init__(
        self,
        classification_head: ClassificationHead,
        text_embedder: Optional[TextEmbedder] = None,
        categorical_variable_net: Optional[CategoricalVariableNet] = None,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            classification_head (ClassificationHead): The classification head module.
            text_embedder (Optional[TextEmbedder]): The text embedding module.
                If not provided, assumes that input text is already embedded (as tensors) and directly passed to the classification head.
            categorical_variable_net (Optional[CategoricalVariableNet]): The categorical variable network module.
                If not provided, assumes no categorical variables are used.
        """
        super().__init__()

        self.text_embedder = text_embedder

        self.categorical_variable_net = categorical_variable_net
        if not self.categorical_variable_net:
            logger.info("🔹 No categorical variable network provided; using only text embeddings.")

        self.classification_head = classification_head

        self._validate_component_connections()

        self.num_classes = self.classification_head.num_classes

        torch.nn.init.zeros_(self.classification_head.net.weight)
        if self.text_embedder is not None:
            self.text_embedder.init_weights()

    def _validate_component_connections(self):
        def _check_text_categorical_connection(self, text_embedder, cat_var_net):
            if cat_var_net.forward_type == CategoricalForwardType.SUM_TO_TEXT:
                if text_embedder.embedding_dim != cat_var_net.output_dim:
                    raise ValueError(
                        "Text embedding dimension must match categorical variable embedding dimension."
                    )
                self.expected_classification_head_input_dim = text_embedder.embedding_dim
            else:
                self.expected_classification_head_input_dim = (
                    text_embedder.embedding_dim + cat_var_net.output_dim
                )

        if self.text_embedder:
            if self.categorical_variable_net:
                _check_text_categorical_connection(
                    self, self.text_embedder, self.categorical_variable_net
                )
            else:
                self.expected_classification_head_input_dim = self.text_embedder.embedding_dim

            if self.expected_classification_head_input_dim != self.classification_head.input_dim:
                raise ValueError(
                    "Classification head input dimension does not match expected dimension from text embedder and categorical variable net."
                )
        else:
            logger.warning(
                "⚠️ No text embedder provided; assuming input text is already embedded or vectorized. Take care that the classification head input dimension matches the input text dimension."
            )

    def forward(
        self,
        input_ids: Annotated[torch.Tensor, "batch seq_len"],
        attention_mask: Annotated[torch.Tensor, "batch seq_len"],
        categorical_vars: Annotated[torch.Tensor, "batch num_cats"],
        **kwargs,
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass implementation.

        Args: output from dataset collate_fn
            input_ids (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text
            attention_mask (torch.Tensor[int]), shape (batch_size, seq_len): Attention mask indicating non-pad tokens
            categorical_vars (torch.Tensor[Long]): Additional categorical features, (batch_size, num_categorical_features)

        Returns:
            torch.Tensor: Model output scores for each class - shape (batch_size, num_classes)
                Raw, not softmaxed.
        """
        encoded_text = input_ids  # clearer name
        if self.text_embedder is None:
            x_text = encoded_text.float()
        else:
            x_text = self.text_embedder(input_ids=encoded_text, attention_mask=attention_mask)

        if self.categorical_variable_net:
            x_cat = self.categorical_variable_net(categorical_vars)

            if (
                self.categorical_variable_net.forward_type
                == CategoricalForwardType.AVERAGE_AND_CONCAT
                or self.categorical_variable_net.forward_type
                == CategoricalForwardType.CONCATENATE_ALL
            ):
                x_combined = torch.cat((x_text, x_cat), dim=1)
            else:
                assert (
                    self.categorical_variable_net.forward_type == CategoricalForwardType.SUM_TO_TEXT
                )
                x_combined = x_text + x_cat
        else:
            x_combined = x_text

        logits = self.classification_head(x_combined)

        return logits
