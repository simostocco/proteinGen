from torch import nn
from .dense_geometry_attention import PrefixEncoder, ModelConfig


class SequenceOnlyActor(nn.Module):
    def __init__(self, c=ModelConfig()):
        super().__init__()
        self.encoder = PrefixEncoder(c, sequence_only=True)
        self.head = nn.Linear(c.width, 20)

    def forward(self, state):
        return self.head(self.encoder(state))
