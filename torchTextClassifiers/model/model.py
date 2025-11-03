"""FastText model components.

This module contains the PyTorch model, Lightning module, and dataset classes
for FastText classification. Consolidates what was previously in pytorch_model.py,
lightning_module.py, and dataset.py.
"""

import logging
from typing import Annotated, List, Union

import torch
from torch import nn

try:
    from captum.attr import LayerIntegratedGradients

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from torchTextClassifiers.model.categorical_var_net import CategoricalVariableNet, ForwardType
from torchTextClassifiers.model.classification_heads import ClassificationHead

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
        embedding_dim: int,
        classification_head: ClassificationHead,
        categorical_variable_net: CategoricalVariableNet = None,
        tokenizer=None,
        num_rows: int = None,
        categorical_vocabulary_sizes: List[int] = None,
        categorical_embedding_dims: Union[List[int], int] = None,
        sparse: bool = False,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            embedding_dim (int): Dimension of the text embedding space.
            buckets (int): Number of rows in the embedding matrix.
            num_classes (int): Number of classes.
            categorical_vocabulary_sizes (List[int]): List of the number of
                modalities for additional categorical features.
            padding_idx (int, optional): Padding index for the text
                descriptions. Defaults to 0.
            sparse (bool): Indicates if Embedding layer is sparse.
        """
        super().__init__()

        if tokenizer is None:
            if num_rows is None:
                raise ValueError(
                    "Either tokenizer or num_rows must be provided (number of rows in the embedding matrix)."
                )
        else:
            if num_rows is not None:
                if num_rows != tokenizer.vocab_size:
                    logger.warning(
                        "num_rows is different from the number of tokens in the tokenizer. Using provided num_rows."
                    )
            else:
                num_rows = tokenizer.vocab_size

        self.num_rows = num_rows
        self.tokenizer = tokenizer
        self.padding_idx = self.tokenizer.padding_idx
        self.embedding_dim = embedding_dim
        self.sparse = sparse

        self.categorical_embedding_dims = categorical_embedding_dims

        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_rows,
            padding_idx=self.padding_idx,
            sparse=sparse,
        )

        self.classification_head = classification_head
        self.categorical_variable_net = categorical_variable_net
        self.num_classes = self.classification_head.num_classes

    def _get_sentence_embedding(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sentence embedding from encoded text.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text

        Returns:
            torch.Tensor: Sentence embeddings, shape (batch_size, embedding_dim)
        """

        # average over non-pad token embeddings. PAD always has 0 vector so no influence in the sum
        # attention mask has 1 for non-pad tokens and 0 for pad token positions
        # TODO: add attention logic at some point
        sentence_embedding = token_embeddings.sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)  # sum is over seq_len dim

        sentence_embedding = torch.nan_to_num(sentence_embedding, 0.0)

        return sentence_embedding

    def forward(
        self,
        encoded_text: Annotated[torch.Tensor, "batch seq_len"],
        attention_mask: Annotated[torch.Tensor, "batch seq_len"],
        categorical_vars: Annotated[torch.Tensor, "batch num_cats"],
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass implementation.

        Args:
            encoded_text (torch.Tensor[Long]), shape (batch_size, seq_len): Tokenized + padded text
            additional_inputs (torch.Tensor[Long]): Additional categorical features, (batch_size, num_categorical_features)

        Returns:
            torch.Tensor: Model output scores for each class
        """

        # Ensure correct dtype and device once
        if encoded_text.dtype != torch.long:
            encoded_text = encoded_text.to(torch.long)

        # Compute embeddings and averaging in a memory-efficient way
        token_embeddings = self.embeddings(encoded_text)  # (batch_size, seq_len, embedding_dim)

        x_text = self._get_sentence_embedding(
            token_embeddings=token_embeddings, attention_mask=attention_mask
        )

        if self.categorical_variable_net:
            x_cat = self.categorical_variable_net(categorical_vars)

            if (
                self.categorical_variable_net.forward_type == ForwardType.AVERAGE_AND_CONCAT
                or self.categorical_variable_net.forward_type == ForwardType.CONCATENATE_ALL
            ):
                x_combined = torch.cat((x_text, x_cat), dim=1)
            else:
                assert self.categorical_variable_net.forward_type == ForwardType.SUM_TO_TEXT
                x_combined = x_text + x_cat

        logits = self.classification_head(x_combined)

        return logits

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
