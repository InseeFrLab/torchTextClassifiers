"""Largely inspired from Andrej Karpathy's nanochat, see here https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

### Some utils used in text_embedder.py for the attention blocks ###


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention

    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


#### Config #####
@dataclass
class AttentionConfig:
    n_layers: int
    n_head: int
    n_kv_head: int
    sequence_len: Optional[int] = None
    positional_encoding: bool = True
    aggregation_method: str = "mean"  # or 'last', or 'first'


#### Attention Block #####

# Composed of SelfAttentionLayer and MLP with residual connections


class Block(nn.Module):
    def __init__(self, config: AttentionConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.attn = SelfAttentionLayer(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


##### Components of the Block #####


class SelfAttentionLayer(nn.Module):
    def __init__(self, config: AttentionConfig, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.apply_positional_encoding = config.positional_encoding

    def forward(self, x, cos_sin=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if self.apply_positional_encoding:
            assert cos_sin is not None, "Rotary embeddings require precomputed cos/sin tensors"
            cos, sin = cos_sin
            q, k = (
                apply_rotary_emb(q, cos, sin),
                apply_rotary_emb(k, cos, sin),
            )  # QK rotary embedding

        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # is_causal=False for non-autoregressive models (BERT-like)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=self.enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
