import math
import torch


def soft_target(reward, done, next_q, log_aa, log_geom, gamma, alpha_aa, alpha_geom):
    if not 0 <= gamma <= 1:
        raise ValueError("discount")
    return torch.where(done, reward, reward + gamma * (next_q - alpha_aa * log_aa - alpha_geom * log_geom))


def monte_carlo_returns(rewards, gamma):
    value = 0.0
    out = []
    for reward in reversed(rewards):
        value = float(reward) + gamma * value
        out.append(value)
    return list(reversed(out))


def n_step_target(rewards, gamma, bootstrap, terminal):
    if not rewards:
        raise ValueError("positive n step")
    return sum(gamma**i * r for i, r in enumerate(rewards)) + (0 if terminal else gamma ** len(rewards) * bootstrap)


def discrete_objective(logits, values, alpha):
    return (torch.softmax(logits, -1) * (alpha * torch.log_softmax(logits, -1) - values)).sum()


def sampled_discrete_objective(logits, index, advantage):
    return -torch.log_softmax(logits, -1)[index] * advantage.detach()


class Temperatures(torch.nn.Module):
    def __init__(self, target_aa, target_geom):
        super().__init__()
        if not 0 <= target_aa <= math.log(20):
            raise ValueError("categorical entropy bounds")
        self.target_aa = target_aa
        self.target_geom = target_geom
        self.log_aa = torch.nn.Parameter(torch.tensor(-2.0))
        self.log_geom = torch.nn.Parameter(torch.tensor(-2.0))

    def loss(self, log_aa, log_geom, geometry_available):
        loss = self.log_aa * (-log_aa.detach() - self.target_aa)
        if geometry_available:
            loss = loss + self.log_geom * (-log_geom.detach() - self.target_geom)
        return loss
