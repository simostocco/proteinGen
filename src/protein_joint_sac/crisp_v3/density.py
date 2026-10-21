"""Physical-unit transformed Gaussian laws; no action clamps in training."""
import math
import torch
from torch.nn import functional as F

def sample(mu,ls,mask,epsilon=None):
 mu=mu.double();ls=ls.double();u=mu+ls.exp()*(torch.randn_like(mu) if epsilon is None else epsilon.double())
 g=torch.stack([u[...,0].exp(),math.pi*u[...,1].sigmoid(),math.pi*u[...,2].tanh()],-1)
 support=(g[...,0]>0)&(g[...,1]>0)&(g[...,1]<math.pi)&(g[...,2].abs()<math.pi)
 if not torch.isfinite(g).all() or not support.all():raise FloatingPointError('training draw reached numerical boundary; no fabricated action')
 return torch.where(mask,g,0.),log_prob(mu,ls,g,mask)
def log_prob(mu,ls,g,mask):
 v=torch.where(mask,g,g.new_tensor([1.,math.pi/2,0.])).double()
 if not ((v[...,0]>0)&(v[...,1]>0)&(v[...,1]<math.pi)&(v[...,2].abs()<math.pi)&torch.isfinite(v).all(-1)).all():raise ValueError('physical action outside open support')
 u=torch.stack([v[...,0].log(),torch.logit(v[...,1]/math.pi),torch.atanh(v[...,2]/math.pi)],-1)
 jac=torch.stack([u[...,0],math.log(math.pi)-F.softplus(-u[...,1])-F.softplus(u[...,1]),math.log(math.pi)+2*(math.log(2)-u[...,2]-F.softplus(-2*u[...,2]))],-1)
 lp=-.5*((u-mu)/ls.exp()).square()-ls-.5*math.log(2*math.pi)-jac
 return torch.where(mask,lp,0.).sum(-1)
def kl(logits,mu,ls,ref,mask):
 lp=logits.log_softmax(-1);rp=ref['logits'].log_softmax(-1);p=lp.exp();a=(p*(lp-rp)).sum(-1)
 # Common invertible transform => physical KL equals latent Gaussian KL.
 g=ref['ls']-ls+.5*((ls.exp()/ref['ls'].exp()).square()+((mu-ref['mu'])/ref['ls'].exp()).square())-.5
 g=torch.where(mask[:,None],g,0.).sum(-1)
 return a,(p*g).sum(-1),g

def semimarkov(reward,discount,duration,terminated,next_value):
 if (duration<1).any():raise ValueError('positive duration required')
 return reward+discount**duration*(~terminated)*next_value
