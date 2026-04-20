"""TextClassification model components.

This module contains the PyTorch model, Lightning module, and dataset classes
for text classification. Consolidates what was previously in pytorch_model.py,
lightning_module.py, and dataset.py.
"""

import logging
from typing import Annotated, Optional, Union

import torch
from torch import nn

from torchTextClassifiers.model.components import (
    CategoricalForwardType,
    CategoricalVariableNet,
    ClassificationHead,
    SentenceEmbedder,
    TokenEmbedder,
)
from torchTextClassifiers.model.components.attention import norm

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
    def __init__(
        self,
        classification_head: ClassificationHead,
        token_embedder: Optional[TokenEmbedder] = None,
        sentence_embedder: Optional[SentenceEmbedder] = None,
        categorical_variable_net: Optional[CategoricalVariableNet] = None,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            classification_head (ClassificationHead): The classification head module.
            token_embedder (Optional[TextEmbedder]): The text embedding module.
                If not provided, assumes that input text is already embedded (as tensors) and directly passed to the classification head.
            sentence_embedder:
            categorical_variable_net (Optional[CategoricalVariableNet]): The categorical variable network module.
                If not provided, assumes no categorical variables are used.
        """
        super().__init__()

        self.token_embedder = token_embedder
        self.sentence_embedder = sentence_embedder

        if self.token_embedder is not None:
            self.token_embedder.init_weights()
            if self.sentence_embedder is None:
                raise ValueError(
                    "You have provided a TokenEmbedder but no SentenceEmbedder: please provide one."
                )

        self.categorical_variable_net = categorical_variable_net
        if not self.categorical_variable_net:
            logger.info("🔹 No categorical variable network provided; using only text embeddings.")

        self.classification_head = classification_head

        self._validate_component_connections()

        torch.nn.init.zeros_(self.classification_head.net.weight)

    def _validate_component_connections(self):
        def _check_text_categorical_connection(self, token_embedder, cat_var_net):
            if cat_var_net.forward_type == CategoricalForwardType.SUM_TO_TEXT:
                if token_embedder.embedding_dim != cat_var_net.output_dim:
                    raise ValueError(
                        "Text embedding dimension must match categorical variable embedding dimension."
                    )
                self.expected_classification_head_input_dim = token_embedder.embedding_dim
            else:
                self.expected_classification_head_input_dim = (
                    token_embedder.embedding_dim + cat_var_net.output_dim
                )

        if self.token_embedder:
            if self.categorical_variable_net:
                _check_text_categorical_connection(
                    self, self.token_embedder, self.categorical_variable_net
                )
            else:
                self.expected_classification_head_input_dim = self.token_embedder.embedding_dim

            if self.expected_classification_head_input_dim != self.classification_head.input_dim:
                raise ValueError(
                    "Classification head input dimension does not match expected dimension from text embedder and categorical variable net."
                )
            if self.sentence_embedder.label_attention_config is not None:
                self.enable_label_attention = True
                if self.classification_head.num_classes != 1:
                    raise ValueError(
                        "Label attention is enabled. TextEmbedder outputs a (num_classes, embedding_dim) tensor, so the ClassificationHead should have an output dimension of 1."
                    )
                # if enable_label_attention is True, label_attention_config exists - and contains num_classes necessarily
                self.num_classes = self.sentence_embedder.label_attention_config.num_classes
            else:
                self.enable_label_attention = False
                self.num_classes = self.classification_head.num_classes
        else:
            logger.warning(
                "⚠️ No text embedder provided; assuming input text is already embedded or vectorized. Take care that the classification head input dimension matches the input text dimension."
            )

    def forward(
        self,
        input_ids: Annotated[torch.Tensor, "batch seq_len"],
        attention_mask: Annotated[torch.Tensor, "batch seq_len"],
        categorical_vars: Annotated[torch.Tensor, "batch num_cats"],
        return_label_attention_matrix: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Memory-efficient forward pass implementation.

        Args: output from dataset collate_fn
            input_ids (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text
            attention_mask (torch.Tensor[int]), shape (batch_size, seq_len): Attention mask indicating non-pad tokens
            categorical_vars (torch.Tensor[Long]): Additional categorical features, (batch_size, num_categorical_features)
            return_label_attention_matrix (bool): If True, returns a dict with logits and label_attention_matrix

        Returns:
            Union[torch.Tensor, dict[str, torch.Tensor]]:
                - If return_label_attention_matrix is False: torch.Tensor of shape (batch_size, num_classes)
                  containing raw logits (not softmaxed)
                - If return_label_attention_matrix is True: dict with keys:
                    - "logits": torch.Tensor of shape (batch_size, num_classes)
                    - "label_attention_matrix": torch.Tensor of shape (batch_size, num_classes, seq_len)
        """
        encoded_text = input_ids  # clearer name
        label_attention_matrix = None
        if self.token_embedder is None:
            x_text = encoded_text.float()
            if return_label_attention_matrix:
                raise ValueError(
                    "return_label_attention_matrix=True requires a token_embedder with label attention enabled"
                )
        else:
            token_embed_output = self.token_embedder(
                input_ids=encoded_text,
                attention_mask=attention_mask,
            )
            x_token = token_embed_output["token_embeddings"]
            sentence_embedding_output = self.sentence_embedder(
                x_token, attention_mask, return_label_attention_matrix=return_label_attention_matrix
            )
            x_text = sentence_embedding_output["sentence_embedding"]
            if return_label_attention_matrix:
                label_attention_matrix = sentence_embedding_output["label_attention_matrix"]

        if self.categorical_variable_net:
            x_cat = self.categorical_variable_net(categorical_vars)

            if self.enable_label_attention:
                # x_text is (batch_size, num_classes, embedding_dim)
                # x_cat is (batch_size, cat_embedding_dim)
                # We need to expand x_cat to (batch_size, num_classes, cat_embedding_dim)
                # x_cat will be appended to x_text along the last dimension for each class
                x_cat = x_cat.unsqueeze(1).expand(-1, self.num_classes, -1)

            if (
                self.categorical_variable_net.forward_type
                == CategoricalForwardType.AVERAGE_AND_CONCAT
                or self.categorical_variable_net.forward_type
                == CategoricalForwardType.CONCATENATE_ALL
            ):
                x_combined = torch.cat((x_text, x_cat), dim=-1)
            else:
                assert (
                    self.categorical_variable_net.forward_type == CategoricalForwardType.SUM_TO_TEXT
                )

                x_combined = x_text + x_cat
        else:
            x_combined = x_text

        logits = self.classification_head(norm(x_combined)).squeeze(-1)

        if return_label_attention_matrix:
            return {"logits": logits, "label_attention_matrix": label_attention_matrix}

        return logits
