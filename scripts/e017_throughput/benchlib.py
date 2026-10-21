from pathlib import Path
import json,copy,time,math,io,statistics,resource,cProfile,pstats,contextlib
from unittest.mock import patch
from collections import defaultdict
import torch
from protein_joint_sac.crisp_v3 import admission as A,optimization as O,networks as N,pipeline as P,fast_execution as X
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha
from protein_joint_sac.crisp_v3.storage import restore,payload,rng_state
from protein_joint_sac.crisp_v3.recovery import state_digest,StoragePolicy,RollingRecovery
from protein_joint_sac.crisp_v3.execution import set_loader
R=Path(__file__).resolve().parents[3];old=R.parent/'e017_crisp_v3_microbatch_20261021_v1';run=old/'training_runs/run_1cc05582586440b78bf2e47183eeeb69'
assert json.load(open(run/'RUN_STATUS.json'))['status']=='PAUSED_OPERATOR'
# Refuse overlap with ANY actual supervised campaign worker, not just an old PID.
for entry in Path('/proc').glob('[0-9]*'):
 try:
  args=(entry/'cmdline').read_bytes().split(b'\0');state=(entry/'stat').read_text().rsplit(') ',1)[1].split()[0]
  if state!='Z' and any(a.endswith(b'/scripts/e017_v3/train.py') for a in args):raise RuntimeError('Campaign worker active: '+entry.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass

manager=RollingRecovery(run,StoragePolicy(),create=False);bundle=manager.current()['latest'];manager._bundle_valid(bundle);assert bundle['update']==372
parent=torch.load(bundle['checkpoint'],map_location='cpu',weights_only=False);config=parent['config'];assert parent['config_sha256']==bundle['config_sha256'];assert not parent['counters'].get('scientific_stop',{}).get('stop')
(R/'results/VERIFIED_PAUSE.json').write_text(json.dumps(dict(update=372,checkpoint=bundle['checkpoint'],checkpoint_sha256=bundle['sha256'],bundle=bundle,all_recovery_fields_present=all(k in parent for k in ['system','optimizers','schedulers','samplers','rng','replay_snapshot','counters'])),indent=2))
torch.use_deterministic_algorithms(True);prior=FrozenPrior('cuda');system=N.System('research').cuda();learner=O.Learner(system,**config['optimizer']);A.configure_initialization(system,learner,config['schedule']);pop=Population();samplers=dict(AA=CoverageSampler(pop),geometry=CoverageSampler(pop),goals=A.InitGoals(pop,1705,json.load(open(config['native_limits']))));inactive=parent['counters']['inactive_model_sha256']
def reset(loader='eager'):
 set_loader(loader);restore(bundle['checkpoint'],system,learner,samplers,config);system.train();pop.cache.clear();pop.reader.small.cache_clear()
 for k,m in system.models.items():
  if k!='actor':m.eval()
 torch.cuda.synchronize()

class Profile:
 def __init__(self):self.times=defaultdict(float);self.calls=defaultdict(int);self.lengths=defaultdict(lambda:dict(decisions=0,residues=0));self.stack=contextlib.ExitStack()
 def wrap(self,name,fn):
  def wrapped(*a,**kw):
   t=time.monotonic()
   try:return fn(*a,**kw)
   finally:self.times[name]+=time.monotonic()-t;self.calls[name]+=1
  return wrapped
 def __enter__(self):
  for obj,name,label in [(pop,'coordinates','coordinate_read_decode'),(pop,'batch','data_batch_total'),(A,'prepare','AA_features'),(X,'prepare','auxiliary_features'),(P,'prepare','original_goal_features'),(FrozenPrior,'__call__','frozen_J1'),(A,'actor_loss','actor_forward_loss'),(X,'loss_with_goal','goal_actor_forward_loss'),(torch.Tensor,'backward','backward'),(O,'finish','clip_adam_zero')]:self.stack.enter_context(patch.object(obj,name,self.wrap(label,getattr(obj,name))))
  fn=pop.decision
  def decision(*a,**k):
   out=fn(*a,**k);v=self.lengths[a[0]+'/'+A.band(len(pop.base.seqs[a[1]]))];v['decisions']+=1;v['residues']+=a[2];return out
  self.stack.enter_context(patch.object(pop,'decision',decision));return self
 def __exit__(self,*a):self.stack.close()
 def result(self):return dict(inclusive_wall_seconds=dict(self.times),calls=dict(self.calls),category_and_length=dict(self.lengths),note='nested asynchronous inclusive clocks, not additive; end-to-end timing synchronizes at update end')

def step(micro=4,fast=False,capture=False,profile=False):
 c=copy.deepcopy(config);c['microbatch']=micro;c['execution_protocol'].update(geometry_microbatch=min(micro,32),goal_microbatch=min(micro,8));captured={};finish=O.finish
 def hook(opt,parameters,clip=1.):
  ps=list(parameters);captured['gradient']=torch.cat([p.grad.flatten().float() if p.grad is not None else torch.zeros_like(p).flatten() for p in ps]).cpu();return finish(opt,ps,clip)
 torch.cuda.reset_peak_memory_stats();t=time.monotonic();prof=Profile()
 with prof if profile else contextlib.nullcontext():
  with patch.object(O,'finish',hook) if capture else contextlib.nullcontext():
   with torch.autocast('cuda',dtype=torch.bfloat16):out=(X.initialization_update if fast else A.initialization_update)(system,learner,prior,pop,samplers['AA'],samplers['geometry'],samplers['goals'],c)
 torch.cuda.synchronize();elapsed=time.monotonic()-t
 row=dict(microbatch=micro,fast=fast,seconds=elapsed,loss=out['loss'],terms=out['terms'],AA_categories=out['AA_categories'],geometry_masks=out['valid_geometry_components'],gradient_norm=out['gradient_norm'],peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),RSS_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,profile=prof.result() if profile else None)
 if capture:captured.update(parameters=torch.cat([p.detach().flatten() for p in system.models['actor'].parameters()]).cpu(),samplers=state_digest({k:v.state_dict() for k,v in samplers.items()}),RNG=state_digest(rng_state()))
 assert math.isfinite(out['loss']) and math.isfinite(out['gradient_norm']);return row,captured

