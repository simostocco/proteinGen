"""Differentiable true-AA-conditioned CA rollouts; native suffix is a loss target only."""
import torch
from contextlib import nullcontext
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..models.batched_bc import encode_batch
from ..models.hybrid_distributions import GeometryDistribution
from ..geometry.ca_builder import next_ca,mask_for_step
VERSION="j1_global_rollout_v1"
def prefix_batch(x,ids,N):
    if len(x)!=len(ids) or x.ndim!=2 or x.shape[-1]!=3:raise ValueError("literal aligned generated prefix")
    return {"x":x[None],"ids":ids[None],"valid":torch.ones(1,len(x),device=x.device,dtype=torch.bool),
            "t":torch.tensor([len(x)],device=x.device),"N":torch.tensor([N],device=x.device)}
def conditional_parameters(model,x,ids,N,next_aa,precision="BF16",recompute=False):
    # Current supervised AA enters only the conditional geometry head, after encoding.
    def forward(x):
        ctx=torch.autocast(x.device.type,dtype=torch.bfloat16) if precision=="BF16" else nullcontext()
        with ctx:
            h=encode_batch(model,prefix_batch(x,ids,N));logits=model.aa_head(h).float()[0]
        with torch.autocast(x.device.type,enabled=False):
            chemical=model.chemistry[next_aa] if model.encoder.config.chemistry else torch.zeros_like(model.chemistry[next_aa])
            p=model.geom(torch.cat([h[0].float(),model.selected_aa.weight[next_aa].float(),chemical.float()]))
        return logits,p
    return checkpoint(forward,x,use_reentrant=False) if recompute and torch.is_grad_enabled() else forward(x)
def radius_of_gyration(x,eps=1e-12):
    return ((x-x.mean(0)).square().sum(-1).mean()+eps).sqrt()
def global_losses(generated,reference,start,clash_radius=2.,separation=3):
    if generated.shape!=reference.shape or not 3<=start<len(generated):raise ValueError("matched fully observed endpoints")
    eps=1e-8
    lr=F.huber_loss(torch.log((radius_of_gyration(generated)+eps)/(radius_of_gyration(reference)+eps)),
                    generated.new_zeros(()),reduction="sum",delta=.5)
    idx=torch.arange(len(generated),device=generated.device)
    mask=(idx[None]-idx[:,None]>=separation)&(idx[None]>=start)
    distances=torch.linalg.vector_norm(generated[:,None]-generated[None,:],dim=-1)
    lc=(F.relu(clash_radius-distances).square()*mask).sum()/(len(generated)-start)
    return lr,lc
def rollout(model,prefix,sequence_ids,N,horizon,reference,precision="BF16",recompute=True,clash_radius=2.):
    start=len(prefix)
    if not 3<=start or start+horizon>N or len(reference)!=start+horizon or len(sequence_ids)!=start+horizon:raise ValueError("qualified prefix and matched native loss endpoint")
    x=prefix;ids=sequence_ids[:start];aa=[];params=[]
    for t in range(start,start+horizon):
        current=sequence_ids[t]
        logits,p=conditional_parameters(model,x,ids,N,current,precision,recompute)
        aa.append(F.cross_entropy(logits[None],current[None]))
        g,_=GeometryDistribution(p[:3].double(),p[3:].double(),mask_for_step(t)).rsample()
        x=torch.cat([x,next_ca(x,g)[None]],0);ids=torch.cat([ids,current[None]],0);params.append(p)
    lr,lc=global_losses(x,reference,start,clash_radius)
    return {"Rg_loss":lr,"clash_loss":lc,"coordinates":x,"generated_prefix_AA_NLL":torch.stack(aa).mean(),
            "geometry_parameters":params,"start":start,"horizon":horizon}
