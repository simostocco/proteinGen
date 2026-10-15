"""Synthetic complete N500 effective-batch2 update; no protein training."""
import os,json,time,resource,torch
from pathlib import Path
from protein_joint_sac.contracts import BuildAction,AA
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
from protein_joint_sac.geometry.local_frames import reuse_static_prefix
from protein_joint_sac.models.build_critic import TwinCritics,reuse_frozen_current
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.rl.checkpointing import load_checked
R=Path(os.environ['E016_ARTIFACT_ROOT']);out=R/'benchmarks/COMPLETE_MICROBATCH_N500.json';assert not out.exists()
torch.set_num_threads(1);torch.manual_seed(1516);torch.backends.cuda.matmul.allow_tf32=False
saved=load_checked(R/'benchmarks/N500_run003/synthetic_step.pt');a=JointActor(ModelConfig(width=128,heads=4,layers=4)).cuda();q=TwinCritics(ModelConfig(width=96,heads=4,layers=3)).cuda();a.load_state_dict(saved['actor']);q.load_state_dict(saved['critics']);alg=HybridSAC(a,q);alg.temperatures.load_state_dict(saved['temperature']);del saved
acts=tuple(BuildAction(AA[i%20],torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64,device='cuda'),mask_for_step(i)) for i in range(500));states=[replay(500,acts[:-2],device='cuda'),replay(500,acts[:-3],device='cuda')];nexts=[replay(500,acts[:-1],device='cuda'),replay(500,acts[:-2],device='cuda')]
rows=[]
def timed(name,fn):
 torch.cuda.synchronize();t=time.monotonic();v=fn();torch.cuda.synchronize();rows.append({'operation':name,'seconds':time.monotonic()-t});return v
targets=[]
with torch.no_grad():
 for i,s in enumerate(nexts):
  with reuse_static_prefix(s),reuse_frozen_current(s):targets.append(s.ca.new_tensor(.3)+.99*timed('exact20_target_'+str(i),lambda:alg.soft_value(s,targets=True)))
def loss(i):
 s=states[i];act=acts[s.t]
 with reuse_static_prefix(s):return (q.q1(s,act)-targets[i]).square()+(q.q2(s,act)-targets[i]).square()
alg.critic_optimizer.zero_grad(set_to_none=True);((loss(0)+loss(1))/2).backward();expected={n:p.grad.detach().cpu().clone() for n,p in q.named_parameters() if p.grad is not None};alg.critic_optimizer.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats();start=time.monotonic()
for i in range(2):timed('critic_microbatch_backward_'+str(i),lambda i=i:(loss(i)/2).backward())
err=0
for n,p in q.named_parameters():
 if n in expected:torch.testing.assert_close(p.grad.cpu(),expected[n],atol=2e-6,rtol=1e-4);err=max(err,float((p.grad.cpu()-expected[n]).abs().max()))
timed('critic_Adam',alg.critic_optimizer.step);alg.critic_optimizer.zero_grad(set_to_none=True);a.zero_grad(set_to_none=True)
for i,s in enumerate(states):torch.manual_seed(2300+i);timed('exact20_actor_microbatch_backward_'+str(i),lambda s=s:alg.backward_actor_exact_streamed(s))
for p in a.parameters():
 if p.grad is not None:p.grad.div_(2)
timed('actor_Adam',alg.actor_optimizer.step)
losses=[]
for s in states:
 _,la,lg=a.sample(s);losses.append(alg.temperatures.loss(la,lg,True))
alg.alpha_optimizer.zero_grad(set_to_none=True);(sum(losses)/2).backward();alg.alpha_optimizer.step();q.polyak(.005)
for model in [a,q,alg.temperatures]:
 for p in model.parameters():assert torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
torch.cuda.synchronize();elapsed=time.monotonic()-start
result={'status':'PASS_COMPLETE_SYNTHETIC_UPDATE','N':500,'effective_batch':2,'live_microbatch':1,'critic_gradient_vs_eager_two_prefix_mean_max_error':err,'actor_gradient_equivalence_reference':'MICROBATCH_N500.json','optimizer_steps':{'critic':1,'actor':1,'temperature':1},'Polyak_updates':1,'measured_update_seconds_excluding_target_construction_and_gradient_reference':elapsed,'throughput_examples_per_second_for_that_interval':2/elapsed,'CUDA_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'CUDA_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'CPU_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'operations':rows,'scope':'batch2 averaged exact categorical SAC objectives; independent continuous MC draws, not production protein training','production_training':False,'AF_calls':0};out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
