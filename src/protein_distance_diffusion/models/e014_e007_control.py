"""Adapter for a future matched control; the E007 implementation is unchanged."""

import torch
from torch import nn

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet


class E007ScaleupControl(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.core = EquivariantPairCoordinateUNet(**config)

    def forward(self, noisy_coordinates, timesteps, lengths, residue_mask):
        continuity = residue_mask[:, :-1] & residue_mask[:, 1:]
        return self.core(noisy_coordinates, timesteps, lengths, residue_mask, continuity)["v_prediction"]

    @torch.no_grad()
    def telemetry(self, noisy_coordinates, timesteps, lengths, residue_mask):
        prediction = self(noisy_coordinates, timesteps, lengths, residue_mask)
        components = []
        for z, n in enumerate(lengths.tolist()):
            features, _, reliability = local_geometry(noisy_coordinates[z, :n])
            components.append(
                dict(
                    global_v=prediction[z, :n],
                    local_effective=torch.zeros_like(prediction[z, :n]),
                    gate=torch.zeros_like(reliability),
                    reliability=reliability,
                    assessable=features[:, 28].bool(),
                )
            )
        return prediction, components
