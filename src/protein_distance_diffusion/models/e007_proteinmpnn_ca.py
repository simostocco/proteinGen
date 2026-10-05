"""Reviewed CA-only ProteinMPNN inference core for E007 Phase 4B.

Mapped from ``protein_mpnn_utils.py`` at ProteinMPNN commit
``8907e6671bfbfc92303b5f79c4b5e6ce47cdef57``. Only the tensor gather helpers,
encoder/decoder layers, CA feature extractor, and deterministic forward path
are retained. Parsing, sampling, training, full-backbone features, and command
entrypoints are intentionally excluded.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PROVENANCE_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
SOURCE_TO_LOCAL = {
    "gather_edges/gather_nodes/cat_neighbors_nodes": "tensor gather helpers",
    "EncLayer/DecLayer/PositionWiseFeedForward": "message-passing layers",
    "PositionalEncodings/CA_ProteinFeatures": "CA-only feature path",
    "ProteinMPNN(ca_only=True).forward": "ProteinMPNNCA.forward",
}


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    flat = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    flat = flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    gathered = torch.gather(nodes, 1, flat)
    return gathered.reshape(*neighbor_idx.shape, nodes.size(2))


def cat_neighbors_nodes(nodes: torch.Tensor, neighbors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.cat([neighbors, gather_nodes(nodes, indices)], dim=-1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden: int, feedforward: int) -> None:
        super().__init__()
        self.W_in = nn.Linear(hidden, feedforward)
        self.W_out = nn.Linear(feedforward, hidden)
        self.act = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.act(self.W_in(values)))


class EncLayer(nn.Module):
    def __init__(self, hidden: int, input_width: int, *, dropout: float = 0.1, scale: int = 30) -> None:
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.W1 = nn.Linear(hidden + input_width, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, hidden)
        self.W11 = nn.Linear(hidden + input_width, hidden)
        self.W12 = nn.Linear(hidden, hidden)
        self.W13 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(hidden, hidden * 4)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        indices: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        attend_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_nodes = cat_neighbors_nodes(nodes, edges, indices)
        expanded = nodes.unsqueeze(-2).expand(-1, -1, edge_nodes.size(-2), -1)
        messages = self.W3(self.act(self.W2(self.act(self.W1(torch.cat([expanded, edge_nodes], dim=-1))))))
        if attend_mask is not None:
            messages = attend_mask.unsqueeze(-1) * messages
        nodes = self.norm1(nodes + self.dropout1(messages.sum(dim=-2) / self.scale))
        nodes = self.norm2(nodes + self.dropout2(self.dense(nodes)))
        if node_mask is not None:
            nodes = node_mask.unsqueeze(-1) * nodes
        edge_nodes = cat_neighbors_nodes(nodes, edges, indices)
        expanded = nodes.unsqueeze(-2).expand(-1, -1, edge_nodes.size(-2), -1)
        messages = self.W13(self.act(self.W12(self.act(self.W11(torch.cat([expanded, edge_nodes], dim=-1))))))
        return nodes, self.norm3(edges + self.dropout3(messages))


class DecLayer(nn.Module):
    def __init__(self, hidden: int, input_width: int, *, dropout: float = 0.1, scale: int = 30) -> None:
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.W1 = nn.Linear(hidden + input_width, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(hidden, hidden * 4)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        attend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded = nodes.unsqueeze(-2).expand(-1, -1, edges.size(-2), -1)
        messages = self.W3(self.act(self.W2(self.act(self.W1(torch.cat([expanded, edges], dim=-1))))))
        if attend_mask is not None:
            messages = attend_mask.unsqueeze(-1) * messages
        nodes = self.norm1(nodes + self.dropout1(messages.sum(dim=-2) / self.scale))
        nodes = self.norm2(nodes + self.dropout2(self.dense(nodes)))
        return node_mask.unsqueeze(-1) * nodes if node_mask is not None else nodes


class PositionalEncodings(nn.Module):
    def __init__(self, embeddings: int, maximum_relative: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = maximum_relative
        self.linear = nn.Linear(2 * maximum_relative + 2, embeddings)

    def forward(self, offset: torch.Tensor, same_chain: torch.Tensor) -> torch.Tensor:
        maximum = self.max_relative_feature
        indices = torch.clamp(offset + maximum, 0, 2 * maximum) * same_chain
        indices = indices + (1 - same_chain) * (2 * maximum + 1)
        return self.linear(F.one_hot(indices, 2 * maximum + 2).float())


class CA_ProteinFeatures(nn.Module):
    def __init__(self, edge_features: int, node_features: int, *, top_k: int, augment_eps: float = 0.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = 16
        self.embeddings = PositionalEncodings(16)
        self.node_embedding = nn.Linear(3, node_features, bias=False)
        self.edge_embedding = nn.Linear(16 + 16 * 9 + 7, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    @staticmethod
    def _quaternions(rotations: torch.Tensor) -> torch.Tensor:
        diagonal = torch.diagonal(rotations, dim1=-2, dim2=-1)
        rxx, ryy, rzz = diagonal.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack([rxx - ryy - rzz, -rxx + ryy - rzz, -rxx - ryy + rzz], dim=-1))
        )
        signs = torch.sign(
            torch.stack(
                [
                    rotations[..., 2, 1] - rotations[..., 1, 2],
                    rotations[..., 0, 2] - rotations[..., 2, 0],
                    rotations[..., 1, 0] - rotations[..., 0, 1],
                ],
                dim=-1,
            )
        )
        scalar = torch.sqrt(F.relu(1 + diagonal.sum(dim=-1, keepdim=True))) / 2
        return F.normalize(torch.cat([signs * magnitudes, scalar], dim=-1), dim=-1)

    def _orientations(self, coordinates: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        differences = coordinates[:, 1:] - coordinates[:, :-1]
        valid = (differences.norm(dim=-1) > 3.6) & (differences.norm(dim=-1) < 4.0)
        unit = F.normalize(differences * valid.unsqueeze(-1), dim=-1)
        u2, u1, u0 = unit[:, :-2], unit[:, 1:-1], unit[:, 2:]
        n2 = F.normalize(torch.cross(u2, u1, dim=-1), dim=-1)
        n1 = F.normalize(torch.cross(u1, u0, dim=-1), dim=-1)
        cosine_angle = torch.clamp(-(u1 * u0).sum(dim=-1), -1 + 1e-6, 1 - 1e-6)
        angle = torch.acos(cosine_angle)
        cosine_dihedral = torch.clamp((n2 * n1).sum(dim=-1), -1 + 1e-6, 1 - 1e-6)
        dihedral = torch.sign((u2 * n1).sum(dim=-1)) * torch.acos(cosine_dihedral)
        nodes = torch.stack(
            [torch.cos(angle), torch.sin(angle) * torch.cos(dihedral), torch.sin(angle) * torch.sin(dihedral)],
            dim=-1,
        )
        nodes = F.pad(nodes, (0, 0, 1, 2))
        first = F.normalize(u2 - u1, dim=-1)
        frames = torch.stack([first, n2, torch.cross(first, n2, dim=-1)], dim=2)
        flattened = F.pad(frames.reshape(*frames.shape[:2], 9), (0, 0, 1, 2))
        neighbor_frames = gather_nodes(flattened, indices).reshape(*indices.shape, 3, 3)
        frames = flattened.reshape(*flattened.shape[:2], 3, 3)
        neighbor_coordinates = gather_nodes(coordinates, indices)
        displacement = neighbor_coordinates - coordinates.unsqueeze(-2)
        local = F.normalize(torch.matmul(frames.unsqueeze(2), displacement.unsqueeze(-1)).squeeze(-1), dim=-1)
        rotations = torch.matmul(frames.unsqueeze(2).transpose(-1, -2), neighbor_frames)
        return nodes, torch.cat([local, self._quaternions(rotations)], dim=-1)

    def _distances(self, coordinates: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        distances = pair_mask * torch.sqrt(
            ((coordinates.unsqueeze(1) - coordinates.unsqueeze(2)) ** 2).sum(dim=-1) + 1e-6
        )
        adjusted = distances + (1 - pair_mask) * distances.max(dim=-1, keepdim=True).values
        values, indices = torch.topk(adjusted, min(self.top_k, coordinates.shape[1]), dim=-1, largest=False)
        return values, indices

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(2.0, 22.0, self.num_rbf, device=distances.device).reshape(1, 1, 1, -1)
        return torch.exp(-(((distances.unsqueeze(-1) - centers) / (20.0 / self.num_rbf)) ** 2))

    def _pair_rbf(self, first: torch.Tensor, second: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        distances = torch.sqrt(((first[:, :, None] - second[:, None]) ** 2).sum(dim=-1) + 1e-6)
        return self._rbf(gather_edges(distances.unsqueeze(-1), indices)[..., 0])

    def forward(
        self,
        coordinates: torch.Tensor,
        mask: torch.Tensor,
        residue_indices: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("ProteinMPNN CA-only input must have shape [batch, length, 3]")
        if self.augment_eps:
            coordinates = coordinates + self.augment_eps * torch.randn_like(coordinates)
        nearest, indices = self._distances(coordinates, mask)
        previous = torch.zeros_like(coordinates)
        following = torch.zeros_like(coordinates)
        previous[:, 1:] = coordinates[:, :-1]
        following[:, :-1] = coordinates[:, 1:]
        nodes, orientation = self._orientations(coordinates, indices)
        positions = (previous, coordinates, following)
        radial = [self._rbf(nearest)]
        radial.extend(
            self._pair_rbf(first, second, indices)
            for first in positions
            for second in positions
            if not (first is coordinates and second is coordinates)
        )
        offset = gather_edges((residue_indices[:, :, None] - residue_indices[:, None, :]).unsqueeze(-1), indices)[
            ..., 0
        ]
        same_chain = gather_edges((chain_labels[:, :, None] == chain_labels[:, None, :]).long().unsqueeze(-1), indices)[
            ..., 0
        ]
        positional = self.embeddings(offset.long(), same_chain)
        edges = self.norm_edges(self.edge_embedding(torch.cat([positional, *radial, orientation], dim=-1)))
        return edges, indices


class ProteinMPNNCA(nn.Module):
    def __init__(self, *, k_neighbors: int = 48) -> None:
        super().__init__()
        hidden = 128
        self.features = CA_ProteinFeatures(hidden, hidden, top_k=k_neighbors, augment_eps=0.0)
        self.W_v = nn.Linear(hidden, hidden)
        self.W_e = nn.Linear(hidden, hidden)
        self.W_s = nn.Embedding(21, hidden)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden, hidden * 2) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden, hidden * 3) for _ in range(3)])
        self.W_out = nn.Linear(hidden, 21)

    def forward(
        self,
        coordinates: torch.Tensor,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        chain_mask: torch.Tensor,
        residue_indices: torch.Tensor,
        chain_labels: torch.Tensor,
        random_values: torch.Tensor,
    ) -> torch.Tensor:
        edges, neighbor_indices = self.features(coordinates, mask, residue_indices, chain_labels)
        nodes = torch.zeros((*edges.shape[:2], edges.shape[-1]), device=edges.device)
        edges = self.W_e(edges)
        attend = mask.unsqueeze(1) * mask.unsqueeze(2)
        attend = gather_edges(attend.unsqueeze(-1), neighbor_indices)[..., 0]
        for layer in self.encoder_layers:
            nodes, edges = layer(nodes, edges, neighbor_indices, mask, attend)
        sequence_edges = cat_neighbors_nodes(self.W_s(sequence), edges, neighbor_indices)
        empty_edges = cat_neighbors_nodes(torch.zeros_like(self.W_s(sequence)), edges, neighbor_indices)
        encoded = cat_neighbors_nodes(nodes, empty_edges, neighbor_indices)
        chain_mask = chain_mask * mask
        order = torch.argsort((chain_mask + 0.0001) * random_values.abs())
        length = neighbor_indices.shape[1]
        permutation = F.one_hot(order, num_classes=length).float()
        backward = torch.einsum(
            "ij,biq,bjp->bqp",
            1 - torch.triu(torch.ones(length, length, device=coordinates.device)),
            permutation,
            permutation,
        )
        attend = torch.gather(backward, 2, neighbor_indices).unsqueeze(-1)
        one_dimensional = mask.reshape(mask.shape[0], mask.shape[1], 1, 1)
        backward_mask = one_dimensional * attend
        forward_encoded = one_dimensional * (1 - attend) * encoded
        for layer in self.decoder_layers:
            decoded = cat_neighbors_nodes(nodes, sequence_edges, neighbor_indices)
            nodes = layer(nodes, backward_mask * decoded + forward_encoded, mask)
        return F.log_softmax(self.W_out(nodes), dim=-1)
