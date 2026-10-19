"""Streaming PDB gradients through the existing certified 40/30/30 loader."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from ..models.batched_bc import encode_batch,losses
from ..data.aa_only_supervised import ExpandedSampler,augment_available

@contextmanager
def supervised_mode(actor):
    old=actor.training;dp=[getattr(b,'dropout_p',0.) for b in actor.encoder.blocks];actor.train()
    for b in actor.encoder.blocks:b.dropout_p=.15
    try:yield
    finally:
        actor.train(old)
        for b,p in zip(actor.encoder.blocks,dp):b.dropout_p=p

class PDBRegularizer:
    """Callbacks accumulate their weighted gradients; they do not zero or step."""
    def __init__(self,actor,data,sampler):self.actor=actor;self.data=data;self.sampler=sampler;self.examples=None
    def begin(self):self.examples=self.sampler.draw();return self.examples
    def backward_AA(self,weight):
        if self.examples is None:raise ValueError('draw paired independent PDB streams once per update')
        total=0.;examples=self.examples[0]
        with supervised_mode(self.actor):
            for start in range(0,64,16):
                b,y,g=self.data.batch(examples[start:start+16],device=next(self.actor.parameters()).device);b=augment_available(b,.075)
                with torch.autocast(next(self.actor.parameters()).device.type,dtype=torch.bfloat16,enabled=next(self.actor.parameters()).device.type=='cuda'):loss=weight*F.cross_entropy(self.actor.aa_head(encode_batch(self.actor,b)).float(),y,reduction='sum')/64
                loss.backward();total+=float(loss.detach())
        return next(self.actor.parameters()).new_tensor(total)
    def backward_geometry(self,weight):
        if self.examples is None:raise ValueError('PDB stream not drawn')
        total=0.;examples=self.examples[1]
        with supervised_mode(self.actor):
            for start in range(0,32,16):
                b,y,g=self.data.batch(examples[start:start+16],device=next(self.actor.parameters()).device);b=augment_available(b,.075)
                with torch.autocast(next(self.actor.parameters()).device.type,dtype=torch.bfloat16,enabled=next(self.actor.parameters()).device.type=='cuda'):_,loss,_=losses(self.actor,b,y,g,geometry_fp32=True)
                value=weight*loss.sum()/32;value.backward();total+=float(value.detach())
        return next(self.actor.parameters()).new_tensor(total)
    def state_dict(self):return {'sampler':self.sampler.state_dict(),'examples':self.examples}
    def load_state_dict(self,state):self.sampler.load_state_dict(state['sampler']);self.examples=state['examples']
