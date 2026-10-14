"""CPU reference update, not a production training controller."""

import torch
from ..contracts import BuildAction, AA
from ..geometry.ca_builder import mask_for_step
from .soft_targets import Temperatures


class HybridSAC:
    def __init__(self, actor, critics, gamma=0.99, lr=1e-3, target_aa=2.0, target_geom=0.0):
        self.actor = actor
        self.critics = critics
        self.gamma = gamma
        self.temperatures = Temperatures(target_aa, target_geom).to(next(actor.parameters()).device)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(list(critics.q1.parameters()) + list(critics.q2.parameters()), lr=lr)
        self.alpha_optimizer = torch.optim.Adam(self.temperatures.parameters(), lr=lr)

    def soft_value(self, state, targets=False, estimator="exact"):
        h = self.actor.encode(state)
        logp = torch.log_softmax(self.actor.aa_head(h), -1)
        p = logp.exp()
        q1, q2 = (self.critics.target1, self.critics.target2) if targets else (self.critics.q1, self.critics.q2)
        aa_alpha = self.temperatures.log_aa.exp().detach()
        g_alpha = self.temperatures.log_geom.exp().detach()
        if estimator not in ["exact", "sampled"]:
            raise ValueError("discrete estimator")
        if estimator == "sampled" and not targets:
            raise ValueError(
                "sampled value is target-only; actor requires exact expectation or explicit score-function estimator"
            )
        indices = range(20) if estimator == "exact" else [int(torch.distributions.Categorical(probs=p).sample())]
        values = []
        for i in indices:
            g, lg = self.actor.geometry_distribution(state, AA[i], h).rsample()
            a = BuildAction(AA[i], g, mask_for_step(state.t))
            v = torch.minimum(q1(state, a), q2(state, a)) - aa_alpha * logp[i] - g_alpha * lg
            values.append(p[i] * v if estimator == "exact" else v)
        return torch.stack(values).sum()

    def actor_loss(self, state):
        return -self.soft_value(state, estimator="exact")

    def update_one(self, state, action, reward, next_state, done):
        with torch.no_grad():
            target = torch.as_tensor(
                reward, dtype=next(self.actor.parameters()).dtype, device=state.ca.device
            ) + self.gamma * (0.0 if done else self.soft_value(next_state, targets=True))
        qloss = (self.critics.q1(state, action) - target).square() + (self.critics.q2(state, action) - target).square()
        self.critic_optimizer.zero_grad()
        qloss.backward()
        self.critic_optimizer.step()
        params = list(self.critics.q1.parameters()) + list(self.critics.q2.parameters())
        for p in params:
            p.requires_grad_(False)
        try:
            aloss = self.actor_loss(state)
            self.actor_optimizer.zero_grad()
            aloss.backward()
            self.actor_optimizer.step()
        finally:
            for p in params:
                p.requires_grad_(True)
        _, la, lg = self.actor.sample(state)
        tloss = self.temperatures.loss(la, lg, any(mask_for_step(state.t)))
        self.alpha_optimizer.zero_grad()
        tloss.backward()
        self.alpha_optimizer.step()
        self.critics.polyak(0.005)
        return {
            "critic_loss": float(qloss.detach()),
            "actor_loss": float(aloss.detach()),
            "temperature_loss": float(tloss.detach()),
            "scope": "CPU_SYNTHETIC_ONLY",
        }

    def backward_actor_exact_streamed(self, state):
        """Same 20-category objective, one Q graph at a time; no optimizer step.

        State is a detached replay observation. Accumulate head/selected-AA gradients
        per category, then propagate the summed tip-gradient through the encoder once.
        """
        if state.ca.requires_grad:
            raise ValueError("SAC replay state must be detached; no backprop through previous actions")
        h = self.actor.encode(state)
        tip = h.detach().requires_grad_(True)
        params = list(self.critics.q1.parameters()) + list(self.critics.q2.parameters())
        old_flags = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        total = 0.0
        try:
            for i, residue in enumerate(AA):
                logp = torch.log_softmax(self.actor.aa_head(tip), -1)
                g, lg = self.actor.geometry_distribution(state, residue, tip).rsample()
                a = BuildAction(residue, g, mask_for_step(state.t))
                q = torch.minimum(self.critics.q1(state, a), self.critics.q2(state, a))
                term = logp[i].exp() * (
                    self.temperatures.log_aa.exp().detach() * logp[i]
                    + self.temperatures.log_geom.exp().detach() * lg
                    - q
                )
                total += float(term.detach())
                term.backward()
            h.backward(tip.grad)
        finally:
            for p, flag in zip(params, old_flags):
                p.requires_grad_(flag)
        return total
