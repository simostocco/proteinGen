"""Frozen single-candidate random-draw monitor with durable attempt commitments."""
import math
from scipy.stats import beta
from core97 import *
N=4096;LOOKS=[0,250,500,1000,2000,3500,5000];ALPHA=.05/(2*2*len(LOOKS))
def interval(success,n=N):
 assert isinstance(success,int) and 0<=success<=n
 return [float(beta.ppf(ALPHA/2,success,n-success+1)) if success else 0.,float(beta.ppf(1-ALPHA/2,success+1,n-success)) if success<n else 1.]
def classify(backbone,hard,n=N,initial_hard=None,KL=0.,diversity=16.,initial_diversity=None):
 if n!=N:return dict(status='INCOMPLETE',optimizer_updates_allowed=False)
 b=interval(int(backbone),n);h=interval(int(hard),n);hfloor=.91875 if initial_hard is None else max(initial_hard,.96875)-.05
 dfloor=12.449118749897715 if initial_diversity is None else .8*max(initial_diversity,15.561398437372143)
 if not all(math.isfinite(v) for v in [KL,diversity]) or KL>1 or diversity<dfloor:status='BREACH'
 elif b[1]<.95 or h[1]<hfloor:status='BREACH'
 elif b[0]>=.95 and h[0]>=hfloor:status='PASS'
 else:status='INCONCLUSIVE'
 return dict(status=status,backbone_CI=b,hard_CI=h,backbone_floor=.95,hard_floor=hfloor,diversity_floor=dfloor,alpha_per_interval=ALPHA,optimizer_updates_allowed=status=='PASS')
class SafetyPause(RuntimeError):pass
class Monitor:
 def __init__(self,path,arm,look,checkpoint_SHA256,actor_fingerprint=None,fixture=False):
  assert arm in ['A','B'] and look in LOOKS
  assert len(checkpoint_SHA256)==64 and all(v in '0123456789abcdef' for v in checkpoint_SHA256)
  self.root=Path(path);assert self.root.resolve().is_relative_to(ROOT.resolve());self.root.mkdir(parents=True,exist_ok=True)
  self.look=look;self.arm=arm;self.fixture=fixture;assert fixture or actor_fingerprint is not None
  self.schedule=read(ROOT/'MONITOR_SCHEDULE.json')['looks'][str(look)]
  self.binding=dict(arm=arm,look=look,checkpoint_SHA256=checkpoint_SHA256,actor_fingerprint=actor_fingerprint,fixture=fixture,source_binding_SHA256=sha(ROOT/'SOURCE_BINDING.json'),monitor_schedule_SHA256=sha(ROOT/'MONITOR_SCHEDULE.json'))
  put(self.root/'BINDING.json',self.binding);self.frontier=0;self.backbone=0;self.hard=0
  for i in range(N):
   cp=self.root/f'commits/{i:05d}.json';rp=self.root/f'rows/{i:05d}.json'
   if not cp.exists():
    if rp.exists():self.commit(i,read(rp))
    else:break
   else:
    seal=read(cp);assert rp.is_file() and sha(rp)==seal['SHA256'];x=read(rp);self.validate(i,x);assert seal['binding']==self.binding;self.count(x)
  cs=list((self.root/'commits').glob('*.json'));assert len(cs)==self.frontier,'NONCONTIGUOUS_MONITOR'
 def validate(self,i,x):
  s=self.schedule[i];assert x['index']==i and x['draw']==s and x['binding']==self.binding
  assert all(type(x[k]) is bool for k in ['complete','aborted','backbone_valid','hard_valid'])
  assert not x['aborted'] or (not x['complete'] and not x['backbone_valid'] and not x['hard_valid'])
  assert x['complete'] or (not x['backbone_valid'] and not x['hard_valid'])
  assert not x['hard_valid'] or x['backbone_valid']
  if not self.fixture:
   row=x['candidate_record'];task=read(ROOT/'MONITOR_POPULATION.json')['tasks'][s['task_index']]
   assert row['seed']==s['candidate_seed'] and row['goal_id']==task['goal']['goal_id']
   assert x['complete']==bool(row['metrics']['complete']) and x['aborted']==(row['termination'] is not None)
   assert x['backbone_valid']==bool(row['metrics']['backbone_valid'] and x['complete'] and not x['aborted'])
   assert x['hard_valid']==bool(row['metrics']['hard_valid'] and x['complete'] and not x['aborted'])
 def count(self,x):self.frontier+=1;self.backbone+=x['backbone_valid'];self.hard+=x['hard_valid']
 def commit(self,i,x,interrupt=None):
  assert i==self.frontier;self.validate(i,x);rp=self.root/f'rows/{i:05d}.json';put(rp,x)
  if interrupt=='after_result':return
  put(self.root/f'commits/{i:05d}.json',dict(SHA256=sha(rp),binding=self.binding));self.count(x)
  put(self.root/'FRONTIER.json',dict(committed=self.frontier,backbone=self.backbone,hard=self.hard),False)
 def decision(self,**kw):return classify(self.backbone,self.hard,self.frontier,**kw)
 def enforce(self,**kw):
  result=self.decision(**kw);evidence=dict(binding=self.binding,committed=self.frontier,decision=result)
  put(self.root/f"decisions/{self.frontier:05d}_{result['status']}.json",evidence)
  put(self.root/'LATEST_DECISION.json',evidence,False)
  if result['status']!='PASS':raise SafetyPause(result['status'])
  return result
 def from_candidate(self,i,row):
  m=row['metrics'];aborted=row['termination'] is not None
  return dict(index=i,draw=self.schedule[i],binding=self.binding,candidate_record=row,complete=bool(m['complete']),aborted=aborted,backbone_valid=bool(m['complete'] and not aborted and m['backbone_valid']),hard_valid=bool(m['complete'] and not aborted and m['hard_valid']))
 def collect(self,P,e,max_new=None):
  # Batch size ONE and fresh world per draw are part of the binomial sampling contract.
  assert not self.fixture
  assert P['c']['state_digest'](e.actor.state_dict())==self.binding['actor_fingerprint']
  e.prior.verify();pop=read(ROOT/'MONITOR_POPULATION.json')['tasks'];count=0
  while self.frontier<N and (max_new is None or count<max_new):
   i=self.frontier;draw=self.schedule[i];task=pop[draw['task_index']]
   row=P['sample'](e,task,'C',1,[draw['candidate_seed']])[0]
   assert row['seed']==draw['candidate_seed'] and row['goal_id']==task['goal']['goal_id']
   self.commit(i,self.from_candidate(i,row));count+=1
  assert P['c']['state_digest'](e.actor.state_dict())==self.binding['actor_fingerprint'];e.prior.verify()
  return self.decision()
