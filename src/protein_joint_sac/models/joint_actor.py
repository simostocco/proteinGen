import torch
from torch import nn
from .dense_geometry_attention import PrefixEncoder, ModelConfig
from .hybrid_distributions import GeometryDistribution
from ..chemistry.reference_adapter import chemistry_table, load
from ..geometry.ca_builder import mask_for_step
from ..contracts import BuildAction


class JointActor(nn.Module):
    def __init__(self, c=ModelConfig()):
        super().__init__()
        self.encoder = PrefixEncoder(c)
        self.aa_head = nn.Linear(c.width, 20)
        self.selected_aa = nn.Embedding(20, c.width)
        self.register_buffer("chemistry", torch.as_tensor(chemistry_table(), dtype=torch.float32))
        self.geom = nn.Sequential(nn.Linear(2 * c.width + 22, c.width), nn.SiLU(), nn.Linear(c.width, 6))

    def encode(self, state):
        return self.encoder(state)

    def logits(self, state):
        return self.aa_head(self.encode(state))

    def geometry_distribution(self, state, residue, h=None):
        h = self.encode(state) if h is None else h
        i = load().INDEX[residue]
        chem = self.chemistry[i] if self.encoder.config.chemistry else torch.zeros_like(self.chemistry[i])
        p = self.geom(torch.cat([h, self.selected_aa.weight[i], chem]))
        return GeometryDistribution(p[:3], p[3:], mask_for_step(state.t))

    def sample(self, state):
        if state.t >= state.N:
            raise ValueError("complete state has no chemical action")
        h = self.encode(state)
        cat = torch.distributions.Categorical(logits=self.aa_head(h))
        i = cat.sample()
        residue = load().AA[int(i)]
        g, logg = self.geometry_distribution(state, residue, h).rsample()
        return BuildAction(residue, g, mask_for_step(state.t)), cat.log_prob(i), logg
