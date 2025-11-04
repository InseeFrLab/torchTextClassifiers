"""FastText model components.

This module contains the PyTorch model, Lightning module, and dataset classes
for FastText classification. Consolidates what was previously in pytorch_model.py,
lightning_module.py, and dataset.py.
"""

import logging
from typing import Annotated, List, Optional

import torch
from torch import nn

try:
    from captum.attr import LayerIntegratedGradients

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

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

    # TODO: move to the wrapper class
    # We should not have anything relating to tokenization in the model class
    # A PyTorch model takes preocessed tensors as input not raw text,
    # and it outputs raw logits, not predictions
    @torch.no_grad()
    def predict(
        self,
        text: List[str],
        categorical_variables: List[List[int]] = None,
        top_k=1,
        explain=False,
    ):
        """
        Args:
            text (List[str]): A list of text observations.
            params (Optional[Dict[str, Any]]): Additional parameters to
                pass to the model for inference.
            top_k (int): for each sentence, return the top_k most likely predictions (default: 1)
            explain (bool): launch gradient integration to have an explanation of the prediction (default: False)
            preprocess (bool): If True, preprocess text. Needs unidecode library.

        Returns:
        if explain is False:
            predictions (torch.Tensor, shape (len(text), top_k)): A tensor containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): A tensor array containing the corresponding confidence scores.
        if explain is True:
            predictions (torch.Tensor, shape (len(text), top_k)): Containing the top_k most likely codes to the query.
            confidence (torch.Tensor, shape (len(text), top_k)): Corresponding confidence scores.
            all_attributions (torch.Tensor, shape (len(text), top_k, seq_len)): A tensor containing the attributions for each token in the text.
            x (torch.Tensor): A tensor containing the token indices of the text.
            id_to_token_dicts (List[Dict[int, str]]): A list of dictionaries mapping token indices to tokens (one for each sentence).
            token_to_id_dicts (List[Dict[str, int]]): A list of dictionaries mapping tokens to token indices: the reverse of those in id_to_token_dicts.
            text (list[str]): A plist containing the preprocessed text (one line for each sentence).
        """

        if explain:
            if not HAS_CAPTUM:
                raise ImportError(
                    "Captum is not installed and is required for explainability. Run 'pip install/uv add torchFastText[explainability]'."
                )
            lig = LayerIntegratedGradients(
                self, self.embeddings
            )  # initialize a Captum layer gradient integrator

        self.eval()

        tokenize_output = self.tokenizer.tokenize(text)

        encoded_text = tokenize_output["input_ids"]  # (batch_size, seq_len)
        attention_mask = tokenize_output["attention_mask"]  # (batch_size, seq_len)

        if categorical_variables is not None:
            categorical_vars = torch.tensor(
                categorical_variables, dtype=torch.float32
            )  # (batch_size, num_categorical_features)
        else:
            categorical_vars = torch.empty((encoded_text.shape[0], 0), dtype=torch.float32)

        pred = self(
            encoded_text, attention_mask, categorical_vars
        )  # forward pass, contains the prediction scores (len(text), num_classes)
        label_scores = pred.detach().cpu()
        label_scores_topk = torch.topk(label_scores, k=top_k, dim=1)

        predictions = label_scores_topk.indices  # get the top_k most likely predictions
        confidence = torch.round(label_scores_topk.values, decimals=2)  # and their scores

        if explain:
            all_attributions = []
            for k in range(top_k):
                attributions = lig.attribute(
                    (encoded_text, attention_mask, categorical_vars),
                    target=torch.Tensor(predictions[:, k]).long(),
                )  # (batch_size, seq_len)
                attributions = attributions.sum(dim=-1)
                all_attributions.append(attributions.detach().cpu())

            all_attributions = torch.stack(all_attributions, dim=1)  # (batch_size, top_k, seq_len)

            return {
                "prediction": predictions,
                "confidence": confidence,
                "attributions": all_attributions,
            }
        else:
            return predictions, confidence
