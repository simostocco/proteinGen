"""GPU exact actor-gradient accumulation, effective batch2, no optimizer update."""
import os,time,json,torch
from pathlib import Path
from protein_joint_sac.contracts import BuildAction,AA
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.rl.checkpointing import load_checked
R=Path(os.environ['E016_ARTIFACT_ROOT']);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
saved=load_checked(R/'benchmarks/N500_run003/synthetic_step.pt')
a=JointActor(ModelConfig(width=128,heads=4,layers=4)).cuda();q=TwinCritics(ModelConfig(width=96,heads=4,layers=3)).cuda();a.load_state_dict(saved['actor']);q.load_state_dict(saved['critics']);alg=HybridSAC(a,q);alg.temperatures.load_state_dict(saved['temperature']);del saved
acts=tuple(BuildAction(AA[i%20],torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64,device='cuda'),mask_for_step(i)) for i in range(500))
states=[replay(500,acts[:-1],device='cuda'),replay(500,acts[:-2],device='cuda')];expected={};individual=[]
for index,s in enumerate(states):
 a.zero_grad(set_to_none=True);torch.manual_seed(2001+index);torch.cuda.synchronize();t=time.monotonic();loss=alg.backward_actor_exact_streamed(s);torch.cuda.synchronize();individual.append(time.monotonic()-t)
 for n,p in a.named_parameters():
  if p.grad is not None:expected[n]=expected.get(n,torch.zeros_like(p.grad,device='cpu'))+p.grad.detach().cpu()/2

a.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
for index,s in enumerate(states):
 torch.manual_seed(2001+index);alg.backward_actor_exact_streamed(s)
for p in a.parameters():
 if p.grad is not None:p.grad.div_(2)
torch.cuda.synchronize();elapsed=time.monotonic()-t;err=0
for n,p in a.named_parameters():
 if n in expected:
  torch.testing.assert_close(p.grad.cpu(),expected[n],atol=2e-6,rtol=1e-4);err=max(err,float((p.grad.cpu()-expected[n]).abs().max()))
result={'status':'PASS','N':500,'effective_batch':2,'live_microbatch':1,'scope':'EXACT_ACTOR_GRADIENT_ACCUMULATION_ONLY','model_actor':[128,4,4],'model_Q':[96,4,3],'individual_backward_seconds':individual,'accumulated_batch2_backward_seconds':elapsed,'examples_per_second':2/elapsed,'max_gradient_error':err,'CUDA_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'CUDA_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'optimizer_steps':0,'production_training':False}
(R/'benchmarks/MICROBATCH_N500.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
