"""One real TRAIN prefix, identical weights; no optimizer step."""
import json
from dataclasses import replace
import torch
import real_pdb_pilot as m
from protein_joint_sac.rl.bc_training import BCConfig,BCTrainer
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.accepted_supervised import AcceptedPopulation
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
c=BCConfig(**json.load(open(m.R/'configs/J2.json')));p=AcceptedPopulation(m.R/'ADMISSION.json','joint',True);gpu=BCTrainer(c,p);gpu.resume(m.R/'runs/J2/checkpoint_012.pt');cpu=BCTrainer(replace(c,device='cpu'),p);cpu.model.load_state_dict(gpu.model.state_dict());i=p.groups['TRAIN'][sorted(p.groups['TRAIN'])[0]][0];a=cpu.loss(i,12)[0];b=gpu.loss(i,12)[0];a.backward();b.backward();error=0.
for x,y in zip(cpu.model.parameters(),gpu.model.parameters()):
 if x.grad is not None:
  if y.grad is None:raise ValueError('gradient-mask mismatch')
  torch.testing.assert_close(x.grad,y.grad.cpu(),atol=1e-4,rtol=1e-3);error=max(error,float((x.grad-y.grad.cpu()).abs().max()))
torch.testing.assert_close(a,b.cpu(),atol=1e-4,rtol=1e-4)
m.write(m.R/'CPU_CUDA_REAL_PREFIX_PARITY.json',{'status':'PASS','sequence_hash':p.rows[i]['sequence_hash'],'prefix':12,'loss_absolute_error':float((a-b.cpu()).abs().detach()),'maximum_gradient_absolute_error':error,'gradient_atol':1e-4,'gradient_rtol':1e-3,'optimizer_steps':0,'scope':'ONE_REAL_TRAIN_PREFIX_FP32_MODELS_FP64_GEOMETRY'})
print('CPU_CUDA_REAL_PREFIX_PARITY_PASS')
