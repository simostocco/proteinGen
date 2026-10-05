"""E005 gated sequence-geometry co-design scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from protein_distance_diffusion.diffusion.gaussian import (
    masked_upper_triangular_loss,
    project_symmetric_zero_diagonal,
)
from protein_distance_diffusion.models.unet import DistanceUNet

E005_ARCHITECTURE_VERSION = "e005_sequence_geometry_codesign_v1"
CONDITIONING_MODES = frozenset(
    {
        "sequence_only",
        "learned_geometry_gating",
        "forced_geometry_conditioning",
    }
)
GATE_ABLATIONS = frozenset(
    {
        "all_learned",
        "all_disabled",
        "sequence_to_geometry_disabled",
        "incoming_geometry_to_sequence_disabled",
        "returned_geometry_to_sequence_disabled",
        "both_geometry_to_sequence_disabled",
        "all_forced_one",
    }
)
GATE_ABLATION_DISABLED_PATHS = {
    "all_learned": frozenset(),
    "all_disabled": frozenset(
        {"incoming_geometry_to_sequence", "sequence_to_geometry", "returned_geometry_to_sequence"}
    ),
    "sequence_to_geometry_disabled": frozenset({"sequence_to_geometry"}),
    "incoming_geometry_to_sequence_disabled": frozenset({"incoming_geometry_to_sequence"}),
    "returned_geometry_to_sequence_disabled": frozenset({"returned_geometry_to_sequence"}),
    "both_geometry_to_sequence_disabled": frozenset({"incoming_geometry_to_sequence", "returned_geometry_to_sequence"}),
    "all_forced_one": frozenset(),
}


@dataclass(frozen=True)
class CoDesignLossWeights:
    """Independent weights for the three E005 objectives."""

    sequence: float = 1.0
    geometry: float = 1.0
    consistency: float = 0.0

    def __post_init__(self) -> None:
        if min(self.sequence, self.geometry, self.consistency) < 0:
            raise ValueError("Co-design loss weights must be non-negative")


def _geometry_residue_features(geometry: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """Summarize one square geometry channel without crossing padded positions."""
    valid = pair_mask[:, 0].to(dtype=geometry.dtype)
    values = geometry[:, 0] * valid
    denominator = valid.sum(dim=-1).clamp_min(1.0)
    mean = values.sum(dim=-1) / denominator
    mean_square = values.square().sum(dim=-1) / denominator
    rms = mean_square.clamp_min(torch.finfo(values.dtype).eps).sqrt()
    residue_mask = pair_mask[:, 0].any(dim=-1).to(dtype=geometry.dtype)
    return torch.stack((mean, rms), dim=-1) * residue_mask[..., None]


class E005SequenceGeometryCoDesign(nn.Module):
    """Couple a canonical-token sequence branch to the validated E004 U-Net."""

    def __init__(
        self,
        *,
        vocabulary_size: int = 22,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        sequence_hidden_dim: int = 128,
        sequence_layers: int = 4,
        sequence_heads: int = 8,
        sequence_feedforward_dim: int = 512,
        sequence_dropout: float = 0.1,
        max_length: int = 500,
        geometry_model: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 2 or not 0 <= pad_token_id < vocabulary_size:
            raise ValueError("Invalid sequence vocabulary configuration")
        if mask_token_id == pad_token_id or not 0 <= mask_token_id < vocabulary_size:
            raise ValueError("MASK must be a distinct vocabulary token")
        if sequence_hidden_dim % sequence_heads:
            raise ValueError("sequence_hidden_dim must be divisible by sequence_heads")
        self.architecture_version = E005_ARCHITECTURE_VERSION
        self.vocabulary_size = int(vocabulary_size)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.max_length = int(max_length)
        self.token_embedding = nn.Embedding(vocabulary_size, sequence_hidden_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, sequence_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=sequence_hidden_dim,
            nhead=sequence_heads,
            dim_feedforward=sequence_feedforward_dim,
            dropout=sequence_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm layers already disable PyTorch's nested-tensor fast path.
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=sequence_layers,
            enable_nested_tensor=False,
        )
        self.sequence_norm = nn.LayerNorm(sequence_hidden_dim)
        self.sequence_logits = nn.Linear(sequence_hidden_dim, vocabulary_size)

        self.geometry_to_sequence = nn.Linear(2, sequence_hidden_dim)
        self.geometry_to_sequence_gate = nn.Linear(sequence_hidden_dim * 2, sequence_hidden_dim)
        self.sequence_to_pair = nn.Linear(sequence_hidden_dim, 1)
        self.sequence_to_pair_gate = nn.Linear(sequence_hidden_dim, 1)
        self.return_geometry_to_sequence = nn.Linear(2, sequence_hidden_dim)
        self.return_geometry_gate = nn.Linear(sequence_hidden_dim * 2, sequence_hidden_dim)

        geometry_config = dict(geometry_model or {})
        geometry_config.setdefault("max_length", max_length)
        self.geometry_model = DistanceUNet(**geometry_config)

    @property
    def downsample_factor(self) -> int:
        return self.geometry_model.downsample_factor

    def _validate_inputs(
        self,
        sequence_token_ids: torch.Tensor,
        residue_mask: torch.Tensor,
        noisy_geometry: torch.Tensor,
        sequence_separation: torch.Tensor,
        pair_mask: torch.Tensor,
        lengths: torch.Tensor,
        geometry_conditioning_mask: torch.Tensor,
        mode: str,
        gate_ablation: str,
    ) -> None:
        if mode not in CONDITIONING_MODES:
            raise ValueError(f"mode must be one of {sorted(CONDITIONING_MODES)}")
        if gate_ablation not in GATE_ABLATIONS:
            raise ValueError(f"gate_ablation must be one of {sorted(GATE_ABLATIONS)}")
        batch, side = sequence_token_ids.shape
        maximum_padded_side = (
            (self.max_length + self.downsample_factor - 1) // self.downsample_factor
        ) * self.downsample_factor
        if side > maximum_padded_side or side % self.downsample_factor:
            raise ValueError("Sequence side exceeds padded max_length or is not divisible by the U-Net factor")
        if residue_mask.shape != (batch, side):
            raise ValueError("residue_mask must have shape [B, L]")
        expected_pair = (batch, 1, side, side)
        if noisy_geometry.shape != expected_pair or sequence_separation.shape != expected_pair:
            raise ValueError("Geometry and separation tensors must have shape [B, 1, L, L]")
        if pair_mask.shape != expected_pair or geometry_conditioning_mask.shape != (batch,):
            raise ValueError("Pair or geometry-conditioning mask has an invalid shape")
        expected_pair_mask = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
        if not torch.equal(pair_mask.bool(), expected_pair_mask):
            raise ValueError("pair_mask must be exactly derived from residue_mask")
        if not torch.equal(residue_mask.sum(dim=1).to(lengths.dtype), lengths):
            raise ValueError("lengths must match residue_mask")
        if sequence_token_ids.min() < 0 or sequence_token_ids.max() >= self.vocabulary_size:
            raise ValueError("sequence_token_ids contain an out-of-range token")

    def _encode_sequence(self, hidden: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.sequence_encoder(hidden, src_key_padding_mask=~residue_mask.bool())
        return self.sequence_norm(encoded) * residue_mask[..., None].to(dtype=encoded.dtype)

    @staticmethod
    def _conditioning_scale(
        mode: str,
        conditioning_mask: torch.Tensor,
        learned_gate: torch.Tensor,
        *,
        gate_path: str,
        gate_ablation: str,
    ) -> torch.Tensor:
        available = conditioning_mask[:, None, None].to(dtype=learned_gate.dtype)
        if mode == "sequence_only":
            return torch.zeros_like(learned_gate)
        if gate_path in GATE_ABLATION_DISABLED_PATHS[gate_ablation]:
            return torch.zeros_like(learned_gate)
        if mode == "forced_geometry_conditioning" or gate_ablation == "all_forced_one":
            return torch.ones_like(learned_gate) * available
        return torch.sigmoid(learned_gate) * available

    def forward(
        self,
        *,
        sequence_token_ids: torch.Tensor,
        residue_mask: torch.Tensor,
        noisy_geometry: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        sequence_separation: torch.Tensor,
        pair_mask: torch.Tensor,
        geometry_conditioning_mask: torch.Tensor,
        mode: str,
        gate_ablation: str = "all_learned",
    ) -> dict[str, torch.Tensor]:
        """Run one recurrent sequence-to-pair-to-sequence feedback cycle."""
        self._validate_inputs(
            sequence_token_ids,
            residue_mask,
            noisy_geometry,
            sequence_separation,
            pair_mask,
            lengths,
            geometry_conditioning_mask,
            mode,
            gate_ablation,
        )
        batch, side = sequence_token_ids.shape
        positions = torch.arange(side, device=sequence_token_ids.device).clamp_max(self.max_length - 1)
        positions = positions[None].expand(batch, -1)
        hidden = self.token_embedding(sequence_token_ids) + self.position_embedding(positions)
        hidden = self._encode_sequence(hidden, residue_mask)

        conditioned_geometry = noisy_geometry
        if mode == "sequence_only":
            conditioned_geometry = torch.zeros_like(noisy_geometry)
        geometry_features = self.geometry_to_sequence(_geometry_residue_features(conditioned_geometry, pair_mask))
        geometry_gate = self._conditioning_scale(
            mode,
            geometry_conditioning_mask,
            self.geometry_to_sequence_gate(torch.cat((hidden, geometry_features), dim=-1)),
            gate_path="incoming_geometry_to_sequence",
            gate_ablation=gate_ablation,
        )
        hidden = self._encode_sequence(hidden + geometry_gate * geometry_features, residue_mask)

        residue_pair_score = self.sequence_to_pair(hidden).squeeze(-1)
        sequence_pair_prediction = 0.5 * (residue_pair_score[:, None, :] + residue_pair_score[:, :, None])
        sequence_pair_prediction = project_symmetric_zero_diagonal(sequence_pair_prediction[:, None], pair_mask)
        pair_gate = self._conditioning_scale(
            mode,
            geometry_conditioning_mask,
            self.sequence_to_pair_gate(hidden),
            gate_path="sequence_to_geometry",
            gate_ablation=gate_ablation,
        ).squeeze(-1)
        symmetric_pair_gate = 0.5 * (pair_gate[:, None, :] + pair_gate[:, :, None])
        geometry_input = conditioned_geometry + sequence_pair_prediction * symmetric_pair_gate[:, None]
        geometry_input = project_symmetric_zero_diagonal(geometry_input, pair_mask)
        geometry_prediction = self.geometry_model(
            geometry_input,
            timesteps,
            lengths,
            sequence_separation,
            pair_mask,
        )

        returned_features = self.return_geometry_to_sequence(_geometry_residue_features(geometry_prediction, pair_mask))
        return_gate = self._conditioning_scale(
            mode,
            geometry_conditioning_mask,
            self.return_geometry_gate(torch.cat((hidden, returned_features), dim=-1)),
            gate_path="returned_geometry_to_sequence",
            gate_ablation=gate_ablation,
        )
        if mode != "sequence_only":
            hidden = self._encode_sequence(hidden + return_gate * returned_features, residue_mask)
        logits = self.sequence_logits(hidden) * residue_mask[..., None].to(dtype=hidden.dtype)
        return {
            "sequence_logits": logits,
            "geometry_prediction": geometry_prediction,
            "sequence_pair_prediction": sequence_pair_prediction,
            "sequence_hidden": hidden,
            "residue_mask": residue_mask.bool(),
            "pair_mask": pair_mask.bool(),
            "geometry_conditioning_mask": geometry_conditioning_mask.bool(),
            "geometry_to_sequence_gate": geometry_gate,
            "sequence_to_geometry_gate": pair_gate,
            "return_geometry_gate": return_gate,
        }


def codesign_losses(
    outputs: dict[str, torch.Tensor],
    *,
    sequence_targets: torch.Tensor,
    masked_token_mask: torch.Tensor,
    geometry_target: torch.Tensor,
    weights: CoDesignLossWeights,
) -> dict[str, torch.Tensor]:
    """Compute separately reported sequence, geometry, and consistency losses."""
    residue_mask = outputs["residue_mask"].bool()
    pair_mask = outputs["pair_mask"].bool()
    valid_tokens = masked_token_mask.bool() & residue_mask
    if not valid_tokens.any():
        raise ValueError("At least one valid masked token is required")
    sequence_loss = F.cross_entropy(
        outputs["sequence_logits"][valid_tokens],
        sequence_targets[valid_tokens],
    )
    geometry_loss = masked_upper_triangular_loss(
        geometry_target.float(),
        outputs["geometry_prediction"].float(),
        pair_mask,
    )
    consistency_loss = masked_upper_triangular_loss(
        outputs["geometry_prediction"].float(),
        outputs["sequence_pair_prediction"].float(),
        pair_mask,
    )
    total = sequence_loss * weights.sequence + geometry_loss * weights.geometry + consistency_loss * weights.consistency
    return {
        "total": total,
        "sequence": sequence_loss,
        "geometry": geometry_loss,
        "consistency": consistency_loss,
    }
