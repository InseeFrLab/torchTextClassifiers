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

from torchTextClassifiers.utilities.checkers import validate_categorical_inputs

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
        num_classes: int,
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

        if isinstance(categorical_embedding_dims, int):
            # if provided categorical embedding dims is an int, average the categorical embeddings before concatenating to sentence embedding
            self.average_cat_embed = True
        else:
            self.average_cat_embed = False

        categorical_vocabulary_sizes, categorical_embedding_dims, num_categorical_features = (
            validate_categorical_inputs(
                categorical_vocabulary_sizes,
                categorical_embedding_dims,
                num_categorical_features=None,
            )
        )

        assert (
            isinstance(categorical_embedding_dims, list) or categorical_embedding_dims is None
        ), "categorical_embedding_dims must be a list of int at this stage"

        if categorical_embedding_dims is None:
            self.average_cat_embed = False

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

        self.num_classes = num_classes
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

        self.categorical_embedding_layers = {}

        # Entry dim for the last layer:
        #   1. embedding_dim if no categorical variables or summing the categrical embeddings to sentence embedding
        #   2. embedding_dim + cat_embedding_dim if averaging the categorical embeddings before concatenating to sentence embedding (categorical_embedding_dims is a int)
        #   3. embedding_dim + sum(categorical_embedding_dims) if concatenating individually the categorical embeddings to sentence embedding (no averaging, categorical_embedding_dims is a list)
        dim_in_last_layer = embedding_dim
        if self.average_cat_embed:
            dim_in_last_layer += categorical_embedding_dims[0]

        if categorical_vocabulary_sizes is not None:
            self.categorical_variables = True
            for var_idx, num_rows in enumerate(categorical_vocabulary_sizes):
                if categorical_embedding_dims is not None:
                    emb = nn.Embedding(
                        embedding_dim=categorical_embedding_dims[var_idx], num_embeddings=num_rows
                    )  # concatenate to sentence embedding
                    if not self.average_cat_embed:
                        dim_in_last_layer += categorical_embedding_dims[var_idx]
                else:
                    emb = nn.Embedding(
                        embedding_dim=embedding_dim, num_embeddings=num_rows
                    )  # sum to sentence embedding
                self.categorical_embedding_layers[var_idx] = emb
                setattr(self, "emb_{}".format(var_idx), emb)
        else:
            self.categorical_variables = False

        self.fc = nn.Linear(dim_in_last_layer, num_classes)

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

        # Handle categorical variables efficiently
        if self.categorical_variables and categorical_vars.numel() > 0:
            cat_embeds = []
            # Process categorical embeddings in batch
            for i, (_, embed_layer) in enumerate(self.categorical_embedding_layers.items()):
                cat_input = categorical_vars[:, i].long()

                # Check if categorical values are within valid range
                vocab_size = embed_layer.num_embeddings
                max_val = cat_input.max().item()
                min_val = cat_input.min().item()

                if max_val >= vocab_size or min_val < 0:
                    raise ValueError(
                        f"Categorical feature {i}: values range [{min_val}, {max_val}] exceed vocabulary size {vocab_size}."
                    )

                cat_embed = embed_layer(cat_input)
                if cat_embed.dim() > 2:
                    cat_embed = cat_embed.squeeze(1)
                cat_embeds.append(cat_embed)

            if self.categorical_embedding_dims is not None:
                if self.average_cat_embed:
                    # Stack and average in one operation
                    x_cat = torch.stack(cat_embeds, dim=0).mean(dim=0)
                    x_combined = torch.cat([x_text, x_cat], dim=1)
                else:
                    # Optimize concatenation
                    x_combined = torch.cat([x_text] + cat_embeds, dim=1)
            else:
                # Sum embeddings efficiently
                x_combined = x_text + torch.stack(cat_embeds, dim=0).sum(dim=0)
        else:
            x_combined = x_text

        # Final linear layer
        return self.fc(x_combined)

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
