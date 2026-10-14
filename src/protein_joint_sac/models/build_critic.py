import copy
import torch
from torch import nn
from .dense_geometry_attention import PrefixEncoder, ModelConfig
from ..geometry.ca_builder import append


class CandidateQ(nn.Module):
    def __init__(self, c=ModelConfig(width=24, heads=3, layers=1)):
        super().__init__()
        self.current = PrefixEncoder(c)
        self.candidate = PrefixEncoder(c)
        self.head = nn.Sequential(nn.Linear(2 * c.width, c.width), nn.SiLU(), nn.Linear(c.width, 1))

    def forward(self, state, action):
        return self.head(torch.cat([self.current(state), self.candidate(append(state, action))])).squeeze(-1)


class TwinCritics(nn.Module):
    def __init__(self, c=ModelConfig(width=24, heads=3, layers=1)):
        super().__init__()
        self.q1 = CandidateQ(c)
        self.q2 = CandidateQ(c)
        self.target1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target2 = copy.deepcopy(self.q2).requires_grad_(False)

    def polyak(self, tau):
        if not 0 <= tau <= 1:
            raise ValueError("Polyak rate")
        with torch.no_grad():
            for online, target in [(self.q1, self.target1), (self.q2, self.target2)]:
                for p, q in zip(online.parameters(), target.parameters()):
                    q.lerp_(p, tau)
