from enum import Enum
from typing import List, Optional, Union

import torch
from torch import nn


class CategoricalForwardType(Enum):
    SUM_TO_TEXT = "EMBEDDING_SUM_TO_TEXT"
    AVERAGE_AND_CONCAT = "EMBEDDING_AVERAGE_AND_CONCAT"
    CONCATENATE_ALL = "EMBEDDING_CONCATENATE_ALL"


class CategoricalVariableNet(nn.Module):
    def __init__(
        self,
        categorical_vocabulary_sizes: List[int],
        categorical_embedding_dims: Optional[Union[List[int], int]] = None,
        text_embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.categorical_vocabulary_sizes = categorical_vocabulary_sizes
        self.categorical_embedding_dims = categorical_embedding_dims
        self.text_embedding_dim = text_embedding_dim

        self._validate_categorical_inputs()
        assert isinstance(
            self.forward_type, CategoricalForwardType
        ), "forward_type must be set after validation"
        assert isinstance(self.output_dim, int), "output_dim must be set as int after validation"

        self.categorical_embedding_layers = {}

        for var_idx, num_rows in enumerate(self.categorical_vocabulary_sizes):
            emb_layer = nn.Embedding(
                num_embeddings=num_rows,
                embedding_dim=self.categorical_embedding_dims[var_idx],
            )
            self.categorical_embedding_layers[var_idx] = emb_layer
            setattr(self, f"categorical_embedding_{var_idx}", emb_layer)

    def forward(self, categorical_vars_tensor: torch.Tensor) -> torch.Tensor:
        cat_embeds = self._get_cat_embeds(categorical_vars_tensor)
        if self.forward_type == CategoricalForwardType.SUM_TO_TEXT:
            x_combined = torch.stack(cat_embeds, dim=0).sum(dim=0)  # (bs, text_embed_dim)
        elif self.forward_type == CategoricalForwardType.AVERAGE_AND_CONCAT:
            x_combined = torch.stack(cat_embeds, dim=0).mean(dim=0)  # (bs, embed_dim)
        elif self.forward_type == CategoricalForwardType.CONCATENATE_ALL:
            x_combined = torch.cat(cat_embeds, dim=1)  # (bs, sum of all cat embed dims)
        else:
            raise ValueError(f"Unknown forward type: {self.forward_type}")

        assert (
            x_combined.dim() == 2
        ), "Output combined tensor must be 2-dimensional (batch_size, embed_dim)"
        assert x_combined.size(1) == self.output_dim

        return x_combined

    def _get_cat_embeds(self, categorical_vars_tensor: torch.Tensor):
        if categorical_vars_tensor.dtype != torch.long:
            categorical_vars_tensor = categorical_vars_tensor.to(torch.long)
        cat_embeds = []

        for i, embed_layer in self.categorical_embedding_layers.items():
            cat_var_tensor = categorical_vars_tensor[:, i]

            # Check if categorical values are within valid range
            vocab_size = embed_layer.num_embeddings
            max_val = cat_var_tensor.max().item()
            min_val = cat_var_tensor.min().item()

            if max_val >= vocab_size or min_val < 0:
                raise ValueError(
                    f"Categorical feature {i}: values range [{min_val}, {max_val}] exceed vocabulary size {vocab_size}."
                )

            cat_embed = embed_layer(cat_var_tensor)
            if cat_embed.dim() > 2:
                cat_embed = cat_embed.squeeze(1)
            cat_embeds.append(cat_embed)

        return cat_embeds

    def _validate_categorical_inputs(self):
        categorical_vocabulary_sizes = self.categorical_vocabulary_sizes
        categorical_embedding_dims = self.categorical_embedding_dims

        if not isinstance(categorical_vocabulary_sizes, list):
            raise TypeError("categorical_vocabulary_sizes must be a list of int")

        if isinstance(categorical_embedding_dims, list):
            if len(categorical_vocabulary_sizes) != len(categorical_embedding_dims):
                raise ValueError(
                    "Categorical vocabulary sizes and their embedding dimensions must have the same length"
                )

        num_categorical_features = len(categorical_vocabulary_sizes)

        # "Transform" embedding dims into a suitable list, or stay None
        if categorical_embedding_dims is not None:
            if isinstance(categorical_embedding_dims, int):
                self.forward_type = CategoricalForwardType.AVERAGE_AND_CONCAT
                self.output_dim = categorical_embedding_dims
                categorical_embedding_dims = [categorical_embedding_dims] * num_categorical_features

            elif isinstance(categorical_embedding_dims, list):
                self.forward_type = CategoricalForwardType.CONCATENATE_ALL
                self.output_dim = sum(categorical_embedding_dims)
            else:
                raise TypeError("categorical_embedding_dims must be an int, a list of int or None")
        else:
            if self.text_embedding_dim is None:
                raise ValueError(
                    "If categorical_embedding_dims is None, text_embedding_dim must be provided"
                )
            self.forward_type = CategoricalForwardType.SUM_TO_TEXT
            self.output_dim = self.text_embedding_dim
            categorical_embedding_dims = [self.text_embedding_dim] * num_categorical_features

        assert (
            isinstance(categorical_embedding_dims, list) or categorical_embedding_dims is None
        ), "categorical_embedding_dims must be a list of int at this point"

        self.categorical_vocabulary_sizes = categorical_vocabulary_sizes
        self.categorical_embedding_dims = categorical_embedding_dims
        self.num_categorical_features = num_categorical_features
