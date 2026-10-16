"""Bounded 12M real-data update/restart preflight, before main lineages."""
import json,time,copy
import torch
import scaled_bc as c
from scaled_train import Engine,cpu_copy,maxdiff,rng_digest
from protein_joint_sac.data.scaled_supervised import Data
c.setup();data=Data(True);configs=json.load(open(c.R/'configs/FROZEN.json'));engine=Engine('J2',data,configs['J2']);first=engine.step();path=c.R/'preflight/restart_update_1.pt';engine.checkpoint(path);second=engine.step();expected=cpu_copy(engine.model.state_dict());opt=cpu_copy(engine.opt.state_dict());sam=copy.deepcopy(engine.sampler.state_dict());rng=rng_digest();restored=Engine('J2',data,configs['J2']);restored.resume(path);actual=restored.step();pe=maxdiff(expected,restored.model.state_dict());oe=maxdiff(opt,restored.opt.state_dict());same=sam==restored.sampler.state_dict() and rng_digest()==rng and second['sample_order_sha256']==actual['sample_order_sha256'];receipt={'status':'PASS' if same and pe<2e-5 and oe<1e-4 else 'FAIL','parameter_error':pe,'optimizer_error':oe,'RNG_sampler_examples_equal':same,'reference_step':second,'reproduced_step':actual,'ceiling_main_updates':0,'engineering_updates':3};c.write(c.R/'preflight/SCALED_RESTART.json',receipt)
if receipt['status']!='PASS':raise RuntimeError('12M restart preflight failed')
print('SCALED_RESTART_PASS',flush=True)
# Paired sampler schedules must match sequence-only/joint through arbitrary microbatching.
from protein_joint_sac.data.scaled_supervised import Sampler
a=Sampler(data);b=Sampler(data);assert a.draw(64)==b.draw(32)+b.draw(32)
# Activation checkpointing measured on identical full-length inputs; no optimizer update.
del engine,restored,opt,expected;torch.cuda.empty_cache();m=c.model('J2');ids=sorted(data.jointids['TRAIN'],key=lambda h:(abs(len(data.seqs[h])-500),h))[:16];examples=[(h,data.owners[h][0],len(data.seqs[h])-1) for h in ids];batch,y,g=c.make(data,examples,True);records=[]
for checkpointed in [False,True]:
 m.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();v,_,_,_=c.loss_call(m,batch,y,g,{'precision':'BF16','activation_checkpoint':checkpointed});v.mean().backward();torch.cuda.synchronize();records.append({'activation_checkpoint':checkpointed,'seconds':time.monotonic()-start,'decisions_per_second':16/(time.monotonic()-start),'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved()})
c.write(c.R/'preflight/ACTIVATION_CHECKPOINT_PROFILE.json',records)
# Config unchanged: best observed uncheckpointed throughput retained; no optimizer capacity reduction.
print('PRETRAIN_ENGINEERING_PASS',flush=True)
