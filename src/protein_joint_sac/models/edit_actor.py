"""Separate full evaluated-sequence interface, never accepted by build encoder."""

import torch
from torch import nn
from ..contracts import EditState, AA


class EditActor(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.emb = nn.Embedding(20, width)
        self.context = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU())
        self.substitute = nn.Linear(width, 20)
        self.stop = nn.Linear(width, 1)

    def forward(self, state):
        if type(state) != EditState:
            raise ValueError("edit requires completed evaluated sequence")
        ids = torch.tensor([AA.index(a) for a in state.protein.sequence], device=self.emb.weight.device)
        N = len(ids)
        conf = state.oracle.plddt
        known = (
            torch.ones(N, device=self.emb.weight.device)
            if conf is not None
            else torch.zeros(N, device=self.emb.weight.device)
        )
        value = (
            torch.tensor(conf, dtype=self.emb.weight.dtype, device=self.emb.weight.device) / 100.0
            if conf is not None
            else torch.zeros(N, device=self.emb.weight.device)
        )
        h = self.context(torch.cat([self.emb(ids), value[:, None], known[:, None]], -1))
        logits = self.substitute(h)
        logits = logits.scatter(1, ids[:, None], float("-inf"))
        if state.remaining_calls == 0:
            logits = torch.full_like(logits, float("-inf"))
        return torch.cat([logits.reshape(-1), self.stop(h.mean(0)).reshape(1)])


class EditCritic(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.emb = nn.Embedding(20, width)
        self.head = nn.Sequential(nn.Linear(width + 3, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, state, action):
        if type(state) != EditState:
            raise ValueError("distinct edit Q schema")
        r = 0 if action.stop else AA.index(action.replacement)
        i = -1 if action.stop else action.position
        return self.head(
            torch.cat(
                [
                    self.emb.weight[r],
                    torch.tensor(
                        [i / state.protein.state.N, state.remaining_calls / 10.0, float(action.stop)],
                        dtype=self.emb.weight.dtype,
                        device=self.emb.weight.device,
                    ),
                ]
            )
        ).squeeze()
