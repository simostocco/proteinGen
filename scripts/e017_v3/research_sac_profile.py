"""One research-capacity hybrid actor step and twin critic steps; engineering only."""
import json,time,resource
import torch
from engineering import R,write,batch
from protein_joint_sac.crisp_v3.prior import FrozenPrior
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.optimization import Learner
start=time.monotonic();torch.manual_seed(1703);prior=FrozenPrior('cuda');system=System('research').cuda();learner=Learner(system);b=batch(prior,length=96);mask=torch.ones(1,3,device='cuda',dtype=torch.bool);torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):
 t=time.monotonic();critic=learner.critic(b,b,torch.tensor([2],device='cuda'),torch.tensor([[3.8,1.8,-.7]],device='cuda',dtype=torch.float64),mask,mask,torch.tensor([.5],device='cuda'),torch.tensor([1],device='cuda'),torch.tensor([False],device='cuda'));torch.cuda.synchronize();critic_seconds=time.monotonic()-t
 t=time.monotonic();actor=learner.low_actor(b,mask);torch.cuda.synchronize();actor_seconds=time.monotonic()-t
prior.verify(learner.optimizers.values());other=sum(p.numel()*8 for k,m in system.models.items() if k not in ['actor','q1','q2'] for p in m.parameters())
write('RESEARCH_STREAMED_SAC_PROFILE.json',dict(status='PASS',fixture='one synthetic prefix N=500,t=96; no quality evaluation',critic=critic,actor=actor,actor_seconds=actor_seconds,twin_critic_seconds=critic_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),unallocated_other_optimizer_moments_bytes=other,conservative_peak_plus_other_moments_bytes=torch.cuda.max_memory_allocated()+other,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_steps=3,J1_updates=0,seconds=time.monotonic()-start));print('research SAC profile PASS',flush=True)
