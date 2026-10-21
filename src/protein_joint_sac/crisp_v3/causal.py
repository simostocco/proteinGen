"""Policy-indexed reachability and common-noise interventions; no future Jacobian fiction."""
from dataclasses import dataclass
import copy
import torch
from .environment import geometric_jacobian,assess

@dataclass(frozen=True)
class ReachabilityOutcome:
 policy_sha256:str
 manager_sha256:str
 goal_sha256:str
 success:bool|None
 terminated:bool
 operationally_censored:bool
 def target(self):
  if not self.terminated or self.operationally_censored or self.success is None:raise ValueError('unresolved reachability, not a failure label')
  return float(self.success)

def intervene(world,geometry_delta,continuation,steps=4):
 """Same saved prefix/RNG; force one physical action perturbation then replay policy.
 continuation(world, forced_delta) returns one step; user-defined policy evaluated
 separately in both branches. This estimates a finite, policy-dependent effect.
 """
 saved=world.export();results=[]
 for delta in [torch.zeros_like(geometry_delta),geometry_delta]:
  world.restore(saved);continuation(world,delta)
  for _ in range(steps-1):
   if world.state.t==world.state.N or world.truncated:break
   continuation(world,None)
  results.append(dict(ca=world.state.ca.detach().clone(),sequence=world.state.residues,events=copy.deepcopy(world.events),complete=world.state.t==world.state.N))
 world.restore(saved);a,b=results
 n=min(len(a['ca']),len(b['ca']));effect=b['ca'][:n]-a['ca'][:n]
 return dict(endpoint_shift=effect[-1],residue_displacements=effect.norm(dim=-1),AA_changes=sum(x!=y for x,y in zip(a['sequence'],b['sequence'])),branches=results,local_J=geometric_jacobian(world.state,world.state.ca.new_tensor([3.8,1.8,.5])))

def influence_features(effect):
 d=effect['residue_displacements'];return torch.stack([d[-1],d.max(),d.square().mean().sqrt(),d.new_tensor(effect['AA_changes'])])

@torch.no_grad()
def subgoal_probabilities(predictor,batch,policy_slots,count=4):
 """Policy-dependent estimates, NOT a certificate. Use only calibrated policy slots.
 Caller may pass these via pipeline.observation(planner_features=...), retaining
 predictor/policy IDs with the observation. Default collection leaves this off.
 """
 out=[]
 for i in range(count):
  b=dict(batch);b['subgoal']=torch.full_like(batch['subgoal'],i);out.append(predictor(b,policy_slots).sigmoid())
 return torch.stack(out,-1)
