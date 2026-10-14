"""Bounded offline diagnostic sampler. Targets never enter actor state."""
import torch
from ..contracts import BuildState,BCPrefixTarget
from ..chemistry.reference_adapter import load
from ..geometry.ca_builder import action_from_positions

def diagnostic_prefix(observation,t,source_sha256):
 m=load();N=len(observation['canonical_full_sequence'])
 p=m.experimental_prefix_for_BC(observation,t=t,requested_N=N)
 ca=torch.tensor([x if x is not None else (0.,0.,0.) for x in p.ca],dtype=torch.float64).reshape(t,3)
 s=BuildState(p.residues,ca,N,torch.tensor([x is not None for x in p.ca],dtype=torch.bool))
 return s

def diagnostic_target(observation,t,source_sha256):
 s=diagnostic_prefix(observation,t,source_sha256)
 nxt=diagnostic_prefix(observation,t+1,source_sha256)
 if not nxt.valid.all():raise ValueError('missing native prefix/next CA; no invented expert geometry')
 action=action_from_positions(nxt.residues[-1],nxt.ca)
 return BCPrefixTarget(s,action,source_sha256,t+1)
