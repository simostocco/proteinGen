"""E006 rich-geometry sequence co-design model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from protein_distance_diffusion.data.rich_geometry import RICH_PAIR_FEATURE_DIM, RICH_RESIDUE_FEATURE_DIM
from protein_distance_diffusion.diffusion.gaussian import masked_upper_triangular_loss, project_symmetric_zero_diagonal
from protein_distance_diffusion.models.codesign import CONDITIONING_MODES
from protein_distance_diffusion.models.unet import DistanceUNet

E006_ARCHITECTURE_VERSION = "e006_rich_geometry_codesign_v1"


@dataclass(frozen=True)
class E006LossWeights:
    sequence: float = 1.0
    geometry: float = 0.1
    consistency: float = 0.01

    def at_step(self, step: int, warmup_steps: int) -> E006LossWeights:
        scale = 1.0 if warmup_steps <= 0 else min(max(step, 0) / warmup_steps, 1.0)
        return E006LossWeights(self.sequence, self.geometry * scale, self.consistency * scale)


class RichGeometryEncoder(nn.Module):
    def __init__(self, residue_width: int, pair_width: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.residue = nn.Sequential(
            nn.Linear(residue_width, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pair = nn.Sequential(
            nn.Linear(pair_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        residue: torch.Tensor,
        pair: torch.Tensor,
        residue_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residue_hidden = self.residue(residue)
        pair_hidden = self.pair(pair) * pair_mask[..., None].to(pair.dtype)
        denominator = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(pair.dtype)
        outgoing = pair_hidden.sum(dim=2) / denominator
        incoming = pair_hidden.sum(dim=1) / denominator
        encoded = self.aggregate(torch.cat((residue_hidden, outgoing, incoming), dim=-1))
        encoded = encoded * residue_mask[..., None].to(encoded.dtype)
        return encoded, pair_hidden


class GeometryConditionedFusion(nn.Module):
    """Gated FiLM residual with capacity proportional to the sequence trunk."""

    def __init__(self, hidden_dim: int, geometry_dim: int, dropout: float) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        geometry: torch.Tensor,
        residue_mask: torch.Tensor,
        conditioning_mask: torch.Tensor,
        *,
        forced: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.condition(geometry).chunk(2, dim=-1)
        learned = torch.sigmoid(self.gate(torch.cat((sequence, geometry), dim=-1)))
        available = conditioning_mask[:, None, None].to(learned.dtype)
        gate = available.expand_as(learned) if forced else learned * available
        update = self.dropout((1.0 + torch.tanh(scale)) * self.norm(sequence) + shift)
        result = (sequence + gate * update) * residue_mask[..., None].to(sequence.dtype)
        return result, gate


class E006RichGeometryCoDesign(nn.Module):
    """E004 geometry U-Net with multi-depth rich geometry-to-sequence fusion."""

    def __init__(
        self,
        *,
        vocabulary_size: int = 22,
        pad_token_id: int = 0,
        sequence_hidden_dim: int = 192,
        sequence_layers: int = 6,
        sequence_heads: int = 8,
        sequence_feedforward_dim: int = 768,
        sequence_dropout: float = 0.1,
        max_length: int = 500,
        rich_hidden_dim: int = 192,
        fusion_layers: tuple[int, ...] | list[int] = (1, 3, 5),
        minimum_fusion_capacity_ratio: float = 0.05,
        minimum_fusion_parameters: int = 100_000,
        geometry_model: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if sequence_hidden_dim % sequence_heads:
            raise ValueError("E006 sequence width must be divisible by attention heads")
        fusion_layers = tuple(int(value) for value in fusion_layers)
        if not fusion_layers or len(set(fusion_layers)) != len(fusion_layers):
            raise ValueError("E006 requires unique multi-depth fusion layers")
        if min(fusion_layers) < 0 or max(fusion_layers) >= sequence_layers:
            raise ValueError("E006 fusion layer index is outside the sequence trunk")
        self.architecture_version = E006_ARCHITECTURE_VERSION
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)
        self.token_embedding = nn.Embedding(vocabulary_size, sequence_hidden_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, sequence_hidden_dim)
        self.sequence_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=sequence_hidden_dim,
                    nhead=sequence_heads,
                    dim_feedforward=sequence_feedforward_dim,
                    dropout=sequence_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(sequence_layers)
            ]
        )
        self.sequence_norm = nn.LayerNorm(sequence_hidden_dim)
        self.rich_encoder = RichGeometryEncoder(
            RICH_RESIDUE_FEATURE_DIM,
            RICH_PAIR_FEATURE_DIM,
            rich_hidden_dim,
            sequence_dropout,
        )
        self.fusion_layers = fusion_layers
        self.fusions = nn.ModuleDict(
            {
                str(index): GeometryConditionedFusion(sequence_hidden_dim, rich_hidden_dim, sequence_dropout)
                for index in fusion_layers
            }
        )
        self.sequence_to_geometry = nn.Sequential(
            nn.Linear(sequence_hidden_dim, rich_hidden_dim),
            nn.GELU(),
            nn.Linear(rich_hidden_dim, 1),
        )
        self.sequence_to_geometry_gate = nn.Linear(sequence_hidden_dim, 1)
        self.sequence_output = nn.Linear(sequence_hidden_dim, vocabulary_size)
        geometry_config = dict(geometry_model or {})
        geometry_config.setdefault("max_length", max_length)
        self.geometry_model = DistanceUNet(**geometry_config)
        counts = self.parameter_counts()
        ratio = counts["geometry_to_sequence_fusion"] / max(counts["sequence_trunk"], 1)
        if counts["geometry_to_sequence_fusion"] < int(minimum_fusion_parameters):
            raise ValueError("E006 geometry-to-sequence fusion is below the reviewed parameter floor")
        if ratio < float(minimum_fusion_capacity_ratio):
            raise ValueError("E006 geometry-to-sequence fusion capacity ratio is below the reviewed threshold")

    @property
    def downsample_factor(self) -> int:
        return self.geometry_model.downsample_factor

    def forward_sequence_pretraining(
        self,
        sequence_token_ids: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the Stage-A sequence path without constructing or using geometry."""
        batch, side = sequence_token_ids.shape
        if residue_mask.shape != (batch, side):
            raise ValueError("E006 Stage-A residue mask shape contradiction")
        positions = torch.arange(side, device=sequence_token_ids.device)[None].expand(batch, -1)
        hidden = self.token_embedding(sequence_token_ids) + self.position_embedding(
            positions.clamp_max(self.max_length - 1)
        )
        for layer in self.sequence_layers:
            hidden = layer(hidden, src_key_padding_mask=~residue_mask.bool())
        hidden = self.sequence_norm(hidden) * residue_mask[..., None].to(hidden.dtype)
        return self.sequence_output(hidden) * residue_mask[..., None].to(hidden.dtype)

    def parameter_counts(self) -> dict[str, int]:
        geometry_output_ids = {id(parameter) for parameter in self.geometry_model.output.parameters()}
        counts = {
            "sequence_trunk": sum(
                parameter.numel()
                for module in (
                    self.token_embedding,
                    self.position_embedding,
                    self.sequence_layers,
                    self.sequence_norm,
                )
                for parameter in module.parameters()
            ),
            "geometry_encoder": sum(parameter.numel() for parameter in self.rich_encoder.parameters()),
            "geometry_to_sequence_fusion": sum(parameter.numel() for parameter in self.fusions.parameters()),
            "sequence_to_geometry_path": sum(
                parameter.numel()
                for module in (self.sequence_to_geometry, self.sequence_to_geometry_gate)
                for parameter in module.parameters()
            ),
            "output_heads": sum(parameter.numel() for parameter in self.sequence_output.parameters())
            + sum(parameter.numel() for parameter in self.geometry_model.output.parameters()),
            "geometry_branch": sum(
                parameter.numel()
                for parameter in self.geometry_model.parameters()
                if id(parameter) not in geometry_output_ids
            ),
        }
        counts["total_model"] = sum(parameter.numel() for parameter in self.parameters())
        return counts

    def forward(
        self,
        *,
        sequence_token_ids: torch.Tensor,
        residue_mask: torch.Tensor,
        rich_residue_features: torch.Tensor,
        rich_pair_features: torch.Tensor,
        pair_feature_mask: torch.Tensor,
        noisy_geometry: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        sequence_separation: torch.Tensor,
        pair_mask: torch.Tensor,
        geometry_conditioning_mask: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        if mode not in CONDITIONING_MODES:
            raise ValueError(f"Unknown E006 conditioning mode: {mode}")
        batch, side = sequence_token_ids.shape
        if residue_mask.shape != (batch, side) or rich_residue_features.shape[:2] != (batch, side):
            raise ValueError("E006 residue feature/mask shape contradiction")
        if rich_pair_features.shape[:3] != (batch, side, side):
            raise ValueError("E006 pair feature shape contradiction")
        expected_pair = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
        if not torch.equal(pair_mask.bool(), expected_pair):
            raise ValueError("E006 biological pair mask contradiction")
        if pair_feature_mask.shape != pair_mask.shape or (pair_feature_mask & ~pair_mask).any():
            raise ValueError("E006 rich pair mask contradiction")
        conditioned = mode != "sequence_only"
        availability = geometry_conditioning_mask.bool() & conditioned
        rich_residue = rich_residue_features if conditioned else torch.zeros_like(rich_residue_features)
        rich_pair = rich_pair_features if conditioned else torch.zeros_like(rich_pair_features)
        geometry_hidden, _pair_hidden = self.rich_encoder(
            rich_residue,
            rich_pair,
            residue_mask,
            pair_feature_mask[:, 0],
        )
        positions = torch.arange(side, device=sequence_token_ids.device)[None].expand(batch, -1)
        hidden = self.token_embedding(sequence_token_ids) + self.position_embedding(
            positions.clamp_max(self.max_length - 1)
        )
        gates = []
        for index, layer in enumerate(self.sequence_layers):
            hidden = layer(hidden, src_key_padding_mask=~residue_mask.bool())
            if str(index) in self.fusions:
                hidden, gate = self.fusions[str(index)](
                    hidden,
                    geometry_hidden,
                    residue_mask,
                    availability,
                    forced=mode == "forced_geometry_conditioning",
                )
                gates.append(gate)
        hidden = self.sequence_norm(hidden) * residue_mask[..., None].to(hidden.dtype)
        logits = self.sequence_output(hidden) * residue_mask[..., None].to(hidden.dtype)
        sequence_residue = self.sequence_to_geometry(hidden).squeeze(-1)
        sequence_pair = 0.5 * (sequence_residue[:, :, None] + sequence_residue[:, None, :])
        pair_gate = torch.sigmoid(self.sequence_to_geometry_gate(hidden)).squeeze(-1)
        if mode == "sequence_only":
            pair_gate = torch.zeros_like(pair_gate)
            geometry_input = torch.zeros_like(noisy_geometry)
        else:
            pair_gate = pair_gate * availability[:, None].to(pair_gate.dtype)
            geometry_input = noisy_geometry
        symmetric_gate = 0.5 * (pair_gate[:, :, None] + pair_gate[:, None, :])
        sequence_pair = project_symmetric_zero_diagonal(sequence_pair[:, None], pair_mask)
        geometry_input = project_symmetric_zero_diagonal(
            geometry_input + sequence_pair * symmetric_gate[:, None],
            pair_mask,
        )
        geometry_prediction = self.geometry_model(
            geometry_input,
            timesteps,
            lengths,
            sequence_separation,
            pair_mask,
        )
        return {
            "sequence_logits": logits,
            "geometry_prediction": geometry_prediction,
            "sequence_pair_prediction": sequence_pair,
            "residue_mask": residue_mask.bool(),
            "pair_mask": pair_mask.bool(),
            "pair_feature_mask": pair_feature_mask.bool(),
            "fusion_gates": torch.stack(gates, dim=1),
            "sequence_to_geometry_gate": pair_gate,
        }


def e006_losses(
    outputs: dict[str, torch.Tensor],
    *,
    sequence_targets: torch.Tensor,
    masked_token_mask: torch.Tensor,
    geometry_target: torch.Tensor,
    weights: E006LossWeights,
) -> dict[str, torch.Tensor]:
    valid_tokens = masked_token_mask.bool() & outputs["residue_mask"]
    if not valid_tokens.any():
        raise ValueError("E006 sequence objective requires masked valid tokens")
    sequence = F.cross_entropy(outputs["sequence_logits"][valid_tokens], sequence_targets[valid_tokens])
    geometry = masked_upper_triangular_loss(
        geometry_target.float(), outputs["geometry_prediction"].float(), outputs["pair_mask"]
    )
    consistency = masked_upper_triangular_loss(
        outputs["geometry_prediction"].float(),
        outputs["sequence_pair_prediction"].float(),
        outputs["pair_mask"],
    )
    weighted = {
        "sequence_weighted": sequence * weights.sequence,
        "geometry_weighted": geometry * weights.geometry,
        "consistency_weighted": consistency * weights.consistency,
    }
    return {
        "sequence": sequence,
        "geometry": geometry,
        "consistency": consistency,
        **weighted,
        "total": sum(weighted.values()),
    }
