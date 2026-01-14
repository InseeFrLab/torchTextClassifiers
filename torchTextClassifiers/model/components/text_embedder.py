import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from torchTextClassifiers.model.components.attention import AttentionConfig, Block, norm


@dataclass
class TextEmbedderConfig:
    vocab_size: int
    embedding_dim: int
    padding_idx: int
    attention_config: Optional[AttentionConfig] = None


class TextEmbedder(nn.Module):
    def __init__(self, text_embedder_config: TextEmbedderConfig):
        super().__init__()

        self.config = text_embedder_config

        self.attention_config = text_embedder_config.attention_config
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)

        if self.attention_config is not None:
            self.attention_config.n_embd = text_embedder_config.embedding_dim

        self.vocab_size = text_embedder_config.vocab_size
        self.embedding_dim = text_embedder_config.embedding_dim
        self.padding_idx = text_embedder_config.padding_idx

        self.embedding_layer = nn.Embedding(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.vocab_size,
            padding_idx=self.padding_idx,
        )

        if self.attention_config is not None:
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Converts input token IDs to their corresponding embeddings."""

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

        text_embedding = self._get_sentence_embedding(
            token_embeddings=token_embeddings, attention_mask=attention_mask
        )

        return text_embedding

    def _get_sentence_embedding(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sentence embedding from embedded tokens - "remove" second dimension.

        Args (output from dataset collate_fn):
            token_embeddings (torch.Tensor[Long]), shape (batch_size, seq_len, embedding_dim): Tokenized + padded text
            attention_mask (torch.Tensor[Long]), shape (batch_size, seq_len): Attention mask indicating non-pad tokens
        Returns:
            torch.Tensor: Sentence embeddings, shape (batch_size, embedding_dim)
        """

        # average over non-pad token embeddings
        # attention mask has 1 for non-pad tokens and 0 for pad token positions

        # mask pad-tokens

        if self.attention_config is not None:
            if self.attention_config.aggregation_method is not None:
                if self.attention_config.aggregation_method == "first":
                    return token_embeddings[:, 0, :]
                elif self.attention_config.aggregation_method == "last":
                    lengths = attention_mask.sum(dim=1).clamp(min=1)  # last non-pad token index + 1
                    return token_embeddings[
                        torch.arange(token_embeddings.size(0)),
                        lengths - 1,
                        :,
                    ]
                else:
                    if self.attention_config.aggregation_method != "mean":
                        raise ValueError(
                            f"Unknown aggregation method: {self.attention_config.aggregation_method}. Supported methods are 'mean', 'first', 'last'."
                        )

        assert self.attention_config is None or self.attention_config.aggregation_method == "mean"

        mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_embeddings = token_embeddings * mask  # (batch_size, seq_len, embedding_dim)

        sentence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(
            min=1.0
        )  # avoid division by zero

        sentence_embedding = torch.nan_to_num(sentence_embedding, 0.0)

        return sentence_embedding

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if out.dim() != 2:
            raise ValueError(
                f"Output of {self.__class__.__name__}.forward must be 2D "
                f"(got shape {tuple(out.shape)})"
            )
        return out

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
