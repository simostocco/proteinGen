"""Minimal reviewed ProGen2 inference architecture for E007 Phase 4B.1.

The implementation is limited to the deterministic causal-language-model
forward required by the pinned ProGen2 checkpoint. It intentionally excludes
sampling, generation helpers, training entrypoints, and remote-code hooks.

The module layout follows the pinned official ProGen2 checkpoint at commit
``9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5`` and preserves its state-dict
names exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

PROVENANCE_COMMIT = "9b4d4fb5ec19c9e55c4bb06305c0e613e46c1cf5"
IMPLEMENTATION_VERSION = "e007_reviewed_progen2_causal_lm_v1"


@dataclass(frozen=True)
class ProGen2Config:
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    rotary_dim: int
    vocab_size: int
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProGen2Config:
        fields = {
            "n_embd",
            "n_head",
            "n_layer",
            "n_positions",
            "rotary_dim",
            "vocab_size",
            "layer_norm_epsilon",
            "activation_function",
            "attn_pdrop",
            "embd_pdrop",
        }
        return cls(**{key: value[key] for key in fields if key in value})


def _rotate_every_two(value: torch.Tensor) -> torch.Tensor:
    first = value[..., ::2]
    second = value[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_rotary(value: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, :, None, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, :, None, :]
    return value * cos + _rotate_every_two(value) * sin


def _rotary_tables(value: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = value.shape[-1]
    frequencies = 1.0 / (
        10_000 ** (torch.arange(0, dimension, 2, device=value.device, dtype=torch.float32) / dimension)
    )
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies).to(value.dtype)
    return angles.sin(), angles.cos()


class ProGen2Attention(nn.Module):
    def __init__(self, config: ProGen2Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("ProGen2 width must be divisible by its head count")
        if config.rotary_dim <= 0 or config.rotary_dim % 2 or config.rotary_dim > config.n_embd // config.n_head:
            raise ValueError("ProGen2 rotary dimension is invalid")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rotary_dim = config.rotary_dim
        self.model_parallel_partitions = 4
        if self.n_head % self.model_parallel_partitions:
            raise ValueError("ProGen2 head count must preserve four-way checkpoint packing")
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        causal = torch.tril(torch.ones(config.n_positions, config.n_positions, dtype=torch.bool))
        self.register_buffer("bias", causal.view(1, 1, config.n_positions, config.n_positions), persistent=True)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=True)

    def _project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _width = hidden.shape
        packed = self.qkv_proj(hidden).reshape(batch, length, self.model_parallel_partitions, -1)
        local_width = self.head_dim * self.n_head // self.model_parallel_partitions
        query, value, key = packed.split(local_width, dim=-1)
        target = (batch, length, self.n_head, self.head_dim)
        return query.reshape(target), key.reshape(target), value.reshape(target)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self._project_qkv(hidden)
        query_rotary, query_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        key_rotary, key_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        sin, cos = _rotary_tables(query_rotary, length)
        query = torch.cat((_apply_rotary(query_rotary, sin, cos), query_pass), dim=-1)
        key = torch.cat((_apply_rotary(key_rotary, sin, cos), key_pass), dim=-1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = self.bias[:, :, :length, :length]
        scores = torch.where(causal, scores, self.masked_bias.to(dtype=scores.dtype))
        if attention_mask is not None:
            valid = attention_mask[:, None, None, :].to(dtype=torch.bool)
            scores = torch.where(valid, scores, self.masked_bias.to(dtype=scores.dtype))
        probabilities = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        attended = torch.matmul(probabilities, value).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.out_proj(attended)


class ProGen2MLP(nn.Module):
    def __init__(self, config: ProGen2Config) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        if config.activation_function != "gelu_new":
            raise ValueError("Pinned ProGen2 checkpoint requires gelu_new")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(hidden), approximate="tanh"))


class ProGen2Block(nn.Module):
    def __init__(self, config: ProGen2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = ProGen2Attention(config)
        self.mlp = ProGen2MLP(config)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.ln_1(hidden)
        return hidden + self.attn(normalized, attention_mask) + self.mlp(normalized)


class ProGen2Transformer(nn.Module):
    def __init__(self, config: ProGen2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([ProGen2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        hidden = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.shape[1] > self.h[0].attn.bias.shape[-1]:
            raise ValueError("ProGen2 input exceeds positional capacity")
        hidden = self.dropout(hidden)
        for block in self.h:
            hidden = block(hidden, attention_mask)
        return self.ln_f(hidden)


class ProGen2ForCausalLM(nn.Module):
    def __init__(self, config: ProGen2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = ProGen2Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden = self.transformer(input_ids, inputs_embeds, attention_mask)
        return SimpleNamespace(logits=self.lm_head(hidden))
