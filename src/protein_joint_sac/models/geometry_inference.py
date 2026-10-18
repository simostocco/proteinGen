"""Inference-only finite-grid support protocol; reference trainable density unchanged."""
import math
import torch
from .hybrid_distributions import GeometryDistribution
PROTOCOL='ca_fp64_interior_v1'

def legacy_action(mu,log_std,mask,epsilon):
    """Original parameter-dtype transform, with explicitly supplied base noise."""
    d=GeometryDistribution(mu,log_std,mask)
    return d.transform(mu+d.log_std.exp()*epsilon.to(mu))

@torch.no_grad()
def corrected_action(mu,log_std,mask,epsilon):
    """Same Gaussian/exp/sigmoid/tanh law evaluated in FP64.

    Rounding to closed endpoints is represented by the nearest interior FP64
    value that also permits a finite inverse. This is an explicit finite-grid
    inference convention, not a new continuous density or a training clamp.
    The latent and representative-action densities are reported separately.
    Nonfinite/underflowed/overflowed distances fail rather than being invented.
    """
    d=GeometryDistribution(mu.double(),log_std.double(),mask)
    u=d.mu+d.log_std.exp()*epsilon.to(d.mu)
    if not torch.isfinite(u).all():raise FloatingPointError('nonfinite geometry latent')
    g=d.transform(u);raw=g.clone();zero=g.new_tensor(0.);pi=g.new_tensor(math.pi)
    if mask[1]:
        # Smallest subnormal normalized fraction remains invertible after pi scaling.
        lo=torch.nextafter(zero,g.new_tensor(1.))*pi
        hi=torch.nextafter(pi,zero)
        g[1]=torch.clamp(g[1],lo,hi)
    if mask[2]:
        hi=torch.nextafter(pi,zero)
        g[2]=torch.clamp(g[2],-hi,hi)
    if not torch.isfinite(g).all() or (mask[0] and not g[0]>0):
        raise FloatingPointError('distance overflow/underflow or nonfinite action; no replacement')
    represented=d.log_prob(g)
    latent=torch.where(d.mask,d.base.log_prob(u)-d.log_jacobian(u),torch.zeros_like(u)).sum()
    if not torch.isfinite(represented) or not torch.isfinite(latent):raise FloatingPointError('nonfinite inverse/density')
    return g,{'protocol':PROTOCOL,'adjusted_components':(g!=raw).cpu().tolist(),
              'latent':u.cpu().tolist(),'latent_action_log_density':float(latent),
              'represented_action_log_density':float(represented),
              'density_note':'Densities differ when finite-grid endpoint representation is repaired; neither is a discrete sampling mass.'}
