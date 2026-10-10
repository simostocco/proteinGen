"""Frozen rank-space timestep bins; checkpointable dedicated random generator."""

import torch

from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion


class SNRBinSampler:
    def __init__(self, seed=15015, bins=8):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        abar = CoordinateVPDiffusion(500).alphas_cumprod
        self.logsnr = torch.log(abar) - torch.log1p(-abar)
        ranked = torch.argsort(self.logsnr, stable=True)
        self.bins = tuple(torch.tensor_split(ranked, bins))
        self.histogram = torch.zeros(bins, dtype=torch.int64)
        self.timestep_histogram = torch.zeros(500, dtype=torch.int64)

    def sample(self, batch, device="cpu", balanced=True, shared_bin=False):
        bin_ids = torch.randint(len(self.bins), (batch,), generator=self.generator)
        if shared_bin:
            bin_ids = bin_ids[:1].expand(batch)
        times = torch.stack(
            [self.bins[b][torch.randint(len(self.bins[b]), (), generator=self.generator)] for b in bin_ids]
        )
        if not balanced:
            times = torch.randint(500, (batch,), generator=self.generator)
            bin_ids = torch.tensor([next(i for i, b in enumerate(self.bins) if int(t) in b.tolist()) for t in times])
        self.histogram += torch.bincount(bin_ids, minlength=len(self.bins))
        self.timestep_histogram += torch.bincount(times, minlength=500)
        return times.to(device), bin_ids.to(device)

    def state_dict(self):
        return {
            "rng": self.generator.get_state(),
            "histogram": self.histogram.clone(),
            "timestep_histogram": self.timestep_histogram.clone(),
            "bins": self.bins,
        }

    def load_state_dict(self, state):
        if len(state["bins"]) != len(self.bins) or any(
            not torch.equal(a, b) for a, b in zip(self.bins, state["bins"], strict=True)
        ):
            raise ValueError("Frozen SNR bin assignment changed")
        self.generator.set_state(state["rng"])
        self.histogram = state["histogram"].clone()
        self.timestep_histogram = state["timestep_histogram"].clone()


def frozen_inverse_gradient_weights(norms):
    norms = torch.as_tensor(norms, dtype=torch.float64)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("All calibrated gradient norms must be positive and finite")
    raw = norms.rsqrt()
    # Solve normalization under clipping, so both the bounds and mean=1 hold.
    low, high = 0.0, float(2 / raw.min())
    for _ in range(100):
        midpoint = (low + high) / 2
        if float((raw * midpoint).clamp(0.5, 2).mean()) < 1:
            low = midpoint
        else:
            high = midpoint
    return (raw * ((low + high) / 2)).clamp(0.5, 2).float()
