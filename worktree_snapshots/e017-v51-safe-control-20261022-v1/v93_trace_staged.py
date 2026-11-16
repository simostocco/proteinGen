"""Likelihood of the complete proposal trace; continuous base variables detached."""
import math
import torch
from torch.nn import functional as F

def base_logprob(mu,logstd,u,mask):
 mu=mu.double();ls=logstd.double().clamp(-5,2);u=u.detach().double();mask=torch.as_tensor(mask,device=mu.device,dtype=torch.bool)
 return torch.where(mask,-.5*((u-mu)/ls.exp()).square()-ls-.5*math.log(2*math.pi),torch.zeros_like(mu)).sum(-1)
def logjac(u,mask):
 u=u.double();j=torch.stack([u[...,0],math.log(math.pi)-F.softplus(-u[...,1])-F.softplus(u[...,1]),math.log(math.pi)+2*(math.log(2)-u[...,2]-F.softplus(-2*u[...,2]))],-1)
 return torch.where(torch.as_tensor(mask,device=u.device,dtype=torch.bool),j,torch.zeros_like(j)).sum(-1)
def loo(values):
 assert values.ndim==1 and len(values)>1
 return (values.sum()-values)/(len(values)-1)
def trace_loss(logp,reference_logp,reward,beta=0.):
 """RLOO unbiased score-function loss for R-beta*KL(full trace||reference).
 All stochastic decisions (including rejected actions) contribute to logp.
 No horizon division, same-group whitening, clipping, or accepted-only shortcut.
 Reference uses exactly the same deterministic filter/fallback/masks.
 """
 llr=logp-reference_logp.detach();augmented=reward.detach()-beta*llr.detach();adv=(augmented-loo(augmented)).detach()
 # Explicit derivative of the KL integrand; its expectation is zero but this
 # term makes the full stochastic-computation derivative transparent.
 return -(adv*logp).mean()+beta*llr.mean()
