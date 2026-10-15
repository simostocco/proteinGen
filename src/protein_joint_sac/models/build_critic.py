import copy
import torch
from torch import nn
from .dense_geometry_attention import PrefixEncoder, ModelConfig
from contextlib import contextmanager
from contextvars import ContextVar
_CURRENT_CACHE=ContextVar("frozen_Q_current_cache",default=None)
@contextmanager
def reuse_frozen_current(state):
    token=_CURRENT_CACHE.set((state,{}))
    try:yield
    finally:_CURRENT_CACHE.reset(token)
from ..geometry.ca_builder import append


class CandidateQ(nn.Module):
    def __init__(self, c=ModelConfig(width=24, heads=3, layers=1)):
        super().__init__()
        self.current = PrefixEncoder(c)
        self.candidate = PrefixEncoder(c)
        self.head = nn.Sequential(nn.Linear(2 * c.width, c.width), nn.SiLU(), nn.Linear(c.width, 1))

    def forward(self, state, action):
        cache=_CURRENT_CACHE.get()
        if cache is not None and cache[0] is state and not state.ca.requires_grad and not any(p.requires_grad for p in self.current.parameters()):
            if id(self) not in cache[1]:cache[1][id(self)]=self.current(state)
            current=cache[1][id(self)]
        else:current=self.current(state)
        return self.head(torch.cat([current, self.candidate(append(state, action))])).squeeze(-1)


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
