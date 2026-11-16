"""Exactly one shared-actor scalar; original AA branch; no categorical mixture."""
import types,sys
import torch
from core97 import ENG
sys.path.insert(0,str(ENG/'source96'));from algorithm import augment
VERSION='TANH_SHARED_GEOMETRY_DELTA_2'
# Parameter units fixed from sum of5000 original scheduled learning rates.
GATE_SCALE=0.00450095
def install(actor):
 assert not hasattr(actor,'burial_gate') and not getattr(actor,'_v96_burial_token_installed',False)
 original=actor.forward
 actor.register_parameter('burial_gate',torch.nn.Parameter(next(actor.parameters()).new_zeros(())))
 def forward(self,batch):
  aa,m0,l0=original(batch)
  _,m1,l1=original(augment(batch))
  gamma=(self.burial_gate/GATE_SCALE).tanh()
  return aa,m0+gamma*(m1-m0),(l0+gamma*(l1-l0)).clamp(-5,2)
 actor.forward=types.MethodType(forward,actor)
 actor._v97_original_forward=original
 actor._v97_gate_version=VERSION
 return actor
