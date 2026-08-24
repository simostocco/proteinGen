"""Decoder-only causal Transformer for p(S | N)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from protein_sequence_generation.embeddings import ContinuousLengthEmbedding, RotaryEmbedding


@dataclass(frozen=True)
class SequenceTransformerConfig:
    """Architecture configuration."""

    vocabulary_size: int = 24
    d_model: int = 384
    num_layers: int = 8
    num_attention_heads: int = 6
    feedforward_dimension: int = 1536
    dropout: float = 0.10
    attention_dropout: float = 0.10
    max_length: int = 500
    activation: str = "gelu"


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and key padding masks."""

    def __init__(self, config: SequenceTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_attention_heads != 0:
            raise ValueError("d_model must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.d_model // config.num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_length=config.max_length)
        self.dropout = config.attention_dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k)
        if hasattr(F, "scaled_dot_product_attention"):
            key_mask = attention_mask[:, None, None, :]
            causal = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            mask = causal[None, None, :, :] & key_mask
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            causal = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            key_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(~(causal[None, None] & key_mask), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.out(attended) * attention_mask[:, :, None].to(x.dtype)


class TransformerBlock(nn.Module):
    """Pre-normalization causal Transformer block with per-block length injection."""

    def __init__(self, config: SequenceTransformerConfig) -> None:
        super().__init__()
        self.length_projection = nn.Linear(config.d_model, config.d_model)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff_norm = nn.LayerNorm(config.d_model)
        activation: nn.Module = nn.GELU() if config.activation == "gelu" else nn.SiLU()
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dimension),
            activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dimension, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, length_embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        conditioned = x + self.length_projection(length_embedding)[:, None, :]
        x = x + self.dropout(self.attn(self.attn_norm(conditioned), attention_mask))
        conditioned = x + self.length_projection(length_embedding)[:, None, :]
        x = x + self.dropout(self.ff(self.ff_norm(conditioned))) * attention_mask[:, :, None].to(x.dtype)
        return x * attention_mask[:, :, None].to(x.dtype)


class ProteinSequenceTransformer(nn.Module):
    """Decoder-only Transformer returning logits over the complete 24-token vocabulary."""

    def __init__(self, config: SequenceTransformerConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = SequenceTransformerConfig(**config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.length_embedding = ContinuousLengthEmbedding(config.d_model, max_length=config.max_length)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocabulary_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return logits shaped [B, Lmax, vocabulary_size]."""
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same [B, L] shape")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError(f"sequence length {input_ids.shape[1]} exceeds max_length={self.config.max_length}")
        h = self.input_dropout(self.token_embedding(input_ids))
        length_embedding = self.length_embedding(lengths)
        h = h * attention_mask[:, :, None].to(h.dtype)
        for block in self.blocks:
            h = block(h, length_embedding, attention_mask)
        h = self.final_norm(h)
        return self.output(h)


def parameter_count(model: nn.Module) -> int:
    """Return trainable plus frozen parameter count."""
    return sum(parameter.numel() for parameter in model.parameters())
