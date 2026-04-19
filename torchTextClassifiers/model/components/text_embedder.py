import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchTextClassifiers.model.components.attention import AttentionConfig, Block, norm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class LabelAttentionConfig:
    n_head: int
    num_classes: int
    embedding_dim: int


@dataclass
class TokenEmbedderConfig:
    vocab_size: int
    embedding_dim: int
    padding_idx: int
    attention_config: Optional[AttentionConfig] = None


@dataclass
class SentenceEmbedderConfig:
    aggregation_method: Optional[str] = "mean"  # or 'last', or 'first'
    label_attention_config: Optional[LabelAttentionConfig] = None


class TokenEmbedder(nn.Module):
    """
    A module that takes tokenized text and outputs dense vector representations (one for each token).

    """

    def __init__(self, token_embedder_config: TokenEmbedderConfig):
        super().__init__()

        self.config = token_embedder_config

        self.attention_config = token_embedder_config.attention_config
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)

        self.vocab_size = token_embedder_config.vocab_size
        self.embedding_dim = token_embedder_config.embedding_dim
        self.padding_idx = token_embedder_config.padding_idx

        self.embedding_layer = nn.Embedding(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.vocab_size,
            padding_idx=self.padding_idx,
        )

        if self.attention_config is not None:
            self.attention_config.n_embd: int = token_embedder_config.embedding_dim
            self.transformer = nn.ModuleDict(
                {
                    "h": nn.ModuleList(
                        [
                            Block(self.attention_config, layer_idx)
                            for layer_idx in range(self.attention_config.n_layers)
                        ]
                    ),
                }
            )

            head_dim = self.attention_config.n_embd // self.attention_config.n_head

            if head_dim * self.attention_config.n_head != self.attention_config.n_embd:
                raise ValueError("embedding_dim must be divisible by n_head.")

            if self.attention_config.positional_encoding:
                if head_dim % 2 != 0:
                    raise ValueError(
                        "embedding_dim / n_head must be even for rotary positional embeddings."
                    )

                if self.attention_config.sequence_len is None:
                    raise ValueError(
                        "sequence_len must be specified in AttentionConfig when positional_encoding is True."
                    )

                self.rotary_seq_len = self.attention_config.sequence_len * 10
                cos, sin = self._precompute_rotary_embeddings(
                    seq_len=self.rotary_seq_len, head_dim=head_dim
                )

                self.register_buffer(
                    "cos", cos, persistent=False
                )  # persistent=False means it's not saved to the checkpoint
                self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)

        # zero out c_proj weights in all blocks
        if self.attention_config is not None:
            for block in self.transformer.h:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            # init the rotary embeddings
            head_dim = self.attention_config.n_embd // self.attention_config.n_head
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.embedding_layer.weight.device.type == "cuda":
            self.embedding_layer.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        encoded_text = input_ids  # clearer name
        if encoded_text.dtype != torch.long:
            encoded_text = encoded_text.to(torch.long)

        batch_size, seq_len = encoded_text.shape
        batch_size_check, seq_len_check = attention_mask.shape

        if batch_size != batch_size_check or seq_len != seq_len_check:
            raise ValueError(
                f"Input IDs and attention mask must have the same batch size and sequence length. "
                f"Got input_ids shape {encoded_text.shape} and attention_mask shape {attention_mask.shape}."
            )

        token_embeddings = self.embedding_layer(
            encoded_text
        )  # (batch_size, seq_len, embedding_dim)

        token_embeddings = norm(token_embeddings)

        if self.attention_config is not None:
            if self.attention_config.positional_encoding:
                cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]
            else:
                cos_sin = None

            for block in self.transformer.h:
                token_embeddings = block(token_embeddings, cos_sin)

            token_embeddings = norm(token_embeddings)

        return {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
        }

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = next(self.parameters()).device

        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting

        return cos, sin


class LabelAttention(nn.Module):
    """
    A head for aggregating token embeddings into label-specific sentence embeddings using cross-attention mechanism.
    Labels are queries that attend over token embeddings (keys and values) to produce label-specific embeddings.

    """

    def __init__(self, label_attention_config: LabelAttentionConfig):
        super().__init__()

        if label_attention_config is None:
            raise ValueError(
                "label_attention_config must be provided to use LabelAttention."
            )

        self.label_attention_config = label_attention_config
        self.num_classes = label_attention_config.num_classes
        self.n_head = label_attention_config.n_head
        self.embedding_dim = label_attention_config.embedding_dim

        # Validate head configuration
        self.head_dim = self.embedding_dim // self.n_head

        if self.head_dim * self.n_head != self.embedding_dim:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by n_head ({self.n_head}). "
                f"Got head_dim = {self.head_dim} with remainder {self.embedding_dim % self.n_head}"
            )

        self.label_embeds = nn.Embedding(self.num_classes, self.embedding_dim)

        self.c_q = nn.Linear(self.embedding_dim, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.embedding_dim, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.embedding_dim, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(
        self,
        token_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
        compute_attention_matrix: Optional[bool] = False,
    ):
        """
        Args:
            token_embeddings (torch.Tensor), shape (batch, seq_len, d_model): Embedded tokens from the text input.
            attention_mask (torch.Tensor, optional), shape (batch, seq_len): Attention mask indicating non-pad tokens (1 for real tokens, 0 for padding).
            compute_attention_matrix (bool): Whether to compute and return the attention matrix.
        Returns:
            dict: {
                "sentence_embedding": torch.Tensor, shape (batch, num_classes, d_model): Label-specific sentence embeddings.
                "attention_matrix": Optional[torch.Tensor], shape (batch, n_head, num_classes, seq_len): Attention weights if compute_attention_matrix is True, else None.
            }

        """
        B, T, C = token_embeddings.size()
        if isinstance(compute_attention_matrix, torch.Tensor):
            compute_attention_matrix = compute_attention_matrix[0].item()
        compute_attention_matrix = bool(compute_attention_matrix)

        # 1. Create label indices [0, 1, ..., C-1] for the whole batch
        label_indices = torch.arange(
            self.num_classes, dtype=torch.long, device=token_embeddings.device
        ).expand(B, -1)

        all_label_embeddings = self.label_embeds(
            label_indices
        )  # Shape: [batch, num_classes, d_model]
        all_label_embeddings = norm(all_label_embeddings)

        q = self.c_q(all_label_embeddings).view(B, self.num_classes, self.n_head, self.head_dim)
        k = self.c_k(token_embeddings).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(token_embeddings).view(B, T, self.n_head, self.head_dim)

        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Prepare attention mask for scaled_dot_product_attention
        # attention_mask: (B, T) with 1 for real tokens, 0 for padding
        # scaled_dot_product_attention expects attn_mask: (B, H, Q, K) or broadcastable shape
        # where True means "mask out" (ignore), False means "attend to"
        attn_mask = None
        if attention_mask is not None:
            # Convert: 0 (padding) -> True (mask out), 1 (real) -> False (attend to)
            attn_mask = attention_mask == 0  # (B, T)
            # Expand to (B, 1, 1, T) for broadcasting across heads and queries
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, self.num_classes, -1)  # (bs, n_labels, d_model)
        y = self.c_proj(y)

        attention_matrix = None
        if compute_attention_matrix:
            # Compute attention scores
            # size (B, n_head, n_labels, seq_len)
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

            # Apply mask to attention scores before softmax
            if attention_mask is not None:
                # attn_mask is already in the right shape: (B, 1, 1, T)
                # We need to apply it to scores of shape (B, n_head, n_labels, T)
                # Set masked positions to -inf so they become 0 after softmax
                attention_scores = attention_scores.masked_fill(attn_mask, float("-inf"))

            attention_matrix = torch.softmax(attention_scores, dim=-1)

        return {"sentence_embedding": y, "attention_matrix": attention_matrix}


class SentenceEmbedder(nn.Module):
    def __init__(self, sentence_embedder_config: SentenceEmbedderConfig):

        """
        A module to aggregate token embeddings.

        Four modes are possible:
        - aggregation_method="mean" (default): token embeddings are averaged
        - aggregation_method="first": sentence embedding is the first token's embedding (commin in BERT-like models ([CLS] token))
        - aggregation_method="last": sentence embedding is the last token's embedding (commin in GPT-like models)
        - aggregation_method=None: in that case you need to provide a label attention
        """

        self.config
        self.label_attention_config = sentence_embedder_config.label_attention_config
        self.aggregation_method = sentence_embedder_config.aggregation_method

        if isinstance(self.label_attention_config, dict):
            self.label_attention_config = LabelAttentionConfig(**self.label_attention_config)
            # Keep self.sentence_embedder_config in sync so downstream components (e.g., LabelAttentionClassifier)
            # always see a LabelAttentionConfig instance rather than a raw dict.
            self.sentence_embedder_config.label_attention_config: LabelAttentionConfig = (
                self.label_attention_config
            )

        if self.label_attention_config is not None:
            self.label_attention_module = LabelAttention(
                label_attention_config=self.label_attention_config
            )
            if self.aggregation_method is not None:
                logger.info(
                    "Warning: aggregation_method is ignored when label_attention_config is provided, since label attention produces label-specific sentence embeddings without further aggregation."
                )
                self.aggregation_method = None  # override to avoid confusion

        if self.aggregation_method not in (None, "mean", "first", "last"):
            raise ValueError(
                f"Unsupported aggregation method: {self.aggregation_method}. Supported methods are None, 'mean', 'first', 'last'."
            )
        if self.aggregation_method is None:
            if self.label_attention_config is None:
                raise ValueError(
                    "aggregation_method cannot be None when label_attention_config is not provided, since we need some way to aggregate token embeddings into a sentence embedding. Please specify an aggregation method (e.g., 'mean') or provide a label_attention_config to use label attention for aggregation."
                )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        return_label_attention_matrix: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute sentence embedding from embedded tokens - "remove" second dimension.

        Args (output from dataset collate_fn):
            token_embeddings (torch.Tensor[Long]), shape (batch_size, seq_len, embedding_dim): Tokenized + padded text
            attention_mask (torch.Tensor[Long]), shape (batch_size, seq_len): Attention mask indicating non-pad tokens
            return_label_attention_matrix (bool): Whether to compute and return the label attention matrix
        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing:
                - 'sentence_embedding': Sentence embeddings, shape (batch_size, embedding_dim) or (batch_size, n_labels, embedding_dim) if label attention is enabled
                - 'label_attention_matrix': Attention matrix if label attention is enabled and return_label_attention_matrix is True, otherwise None
        """
        if self.aggregation_method is not None:  # default is "mean"
            if self.aggregation_method == "first":
                return {
                    "sentence_embedding": token_embeddings[:, 0, :],
                    "label_attention_matrix": None,
                }
            elif self.aggregation_method == "last":
                lengths = attention_mask.sum(dim=1).clamp(min=1)  # last non-pad token index + 1
                return {
                    "sentence_embedding": token_embeddings[
                        torch.arange(token_embeddings.size(0)),
                        lengths - 1,
                        :,
                    ],
                    "label_attention_matrix": None,
                }
            else:  # mean
                mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                masked_embeddings = token_embeddings * mask  # (batch_size, seq_len, embedding_dim)
                sentence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(
                    min=1.0
                )  # avoid division by zero

                sentence_embedding = torch.nan_to_num(sentence_embedding, 0.0)
                return {
                    "sentence_embedding": sentence_embedding,
                    "label_attention_matrix": None,
                }

        else:
            label_attention_result = self.label_attention_module(
                token_embeddings,
                attention_mask=attention_mask,
                compute_attention_matrix=return_label_attention_matrix,
            )
            sentence_embedding = label_attention_result[
                "sentence_embedding"
            ]  # (bs, n_labels, d_embed), so classifier needs to be a (d_embed, 1) matrix
            label_attention_matrix = label_attention_result["attention_matrix"]
            return {
                "sentence_embedding": sentence_embedding,
                "label_attention_matrix": label_attention_matrix,
            }
