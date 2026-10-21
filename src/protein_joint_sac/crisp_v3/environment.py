"""Native-calibrated geometric task environment. Not a biological quality oracle."""
from dataclasses import dataclass,asdict
import copy,math
import numpy as np
import torch
from protein_joint_sac.contracts import BuildState,BuildAction,AA
from protein_joint_sac.geometry.ca_builder import append,mask_for_step,next_ca
from protein_joint_sac.models.geometry_inference import corrected_action
from .density import log_prob

REWARD_PROTOCOL='crisp_v3_terminal_geometric_v1'

def frame(x):
 a=x[-1]-x[-2];a=a/torch.linalg.vector_norm(a);n=torch.linalg.cross(x[-2]-x[-3],a);n=n/torch.linalg.vector_norm(n);return torch.stack([a,torch.linalg.cross(n,a),n])
def goal_tokens(task,device):
 goals=[]
 for i,p in enumerate(task.get('targets',[])):
  r=torch.zeros(36,device=device);r[:3]=torch.tensor(p['position'],device=device)/50;r[32]=p['index']/task['N'];r[33]=1
  if p.get('orientation') is not None:r[3:12]=torch.tensor(p['orientation'],device=device).flatten();r[34]=1
  goals.append(r)
 for index,aa in sorted(task.get('motifs',{}).items(),key=lambda x:int(x[0])):
  r=torch.zeros(36,device=device);r[12+AA.index(aa)]=1;r[32]=int(index)/task['N'];r[35]=1;goals.append(r)
 if not goals:goals=[torch.zeros(36,device=device)]
 return torch.stack(goals)[None],torch.tensor([[bool(r[33:].any()) for r in goals]],device=device)

def assess(state,task,limits):
 if state.t!=state.N:return dict(success=False,complete=False)
 x=state.ca;d=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1);finite=bool(torch.isfinite(x).all());rg=torch.sqrt(((x-x.mean(0))**2).sum(-1).mean())/state.N**(1/3);dist=torch.cdist(x,x);ix=torch.arange(state.N,device=x.device);pairs=(ix[:,None]-ix[None,:]>=3);clash=float(((dist<2)&pairs).sum())*100/state.N
 nondegenerate=all(float(torch.linalg.vector_norm(torch.linalg.cross(x[i]-x[i-1],x[i-1]-x[i-2])))>1e-8 for i in range(2,len(x)))
 valid=finite and bool(d.min()>=limits['adjacent_low_A'] and d.max()<=limits['adjacent_high_A']) and clash<=limits['clashes_per100_max'] and limits['Rg_normalized_low']<=float(rg)<=limits['Rg_normalized_high'] and nondegenerate
 errors=[];ok=True
 for p in task.get('targets',[]):
  i=p['index'];e=float(torch.linalg.vector_norm(x[i]-x.new_tensor(p['position'])));ok&=e<=p['tolerance_A'];o=None
  if p.get('orientation') is not None:
   if i<2:raise ValueError('orientation requires three anchors')
   rot=frame(x[:i+1])@x.new_tensor(p['orientation']).T;o=math.degrees(math.acos(float(((rot.trace()-1)/2).clamp(-1,1))));ok&=o<=p['orientation_tolerance_deg']
  errors.append(dict(index=i,position_error_A=e,orientation_error_deg=o))
 motifs=all(state.residues[int(i)]==a for i,a in task.get('motifs',{}).items())
 return dict(success=bool(valid and motifs and ok),complete=True,valid=bool(valid),motifs=motifs,errors=errors,clashes_per100=clash,Rg_normalized=float(rg))
def potential(state,task):
 future=[p for p in task.get('targets',[]) if p['index']>=state.t]
 if not future:return 0.
 return -float(np.mean([min(1,max(0,float(torch.linalg.vector_norm(state.ca[-1]-state.ca.new_tensor(p['position'])))-p['tolerance_A'])/(3.8*max(1,p['index']-state.t+1))) for p in future]))
def reachable_bound(state,target,max_bond):
 # Necessary geometric contour bound only, not a sufficient learned reachability claim.
 return float(torch.linalg.vector_norm(state.ca[-1]-state.ca.new_tensor(target['position'])))<=max_bond*max(0,target['index']-state.t+1)+target['tolerance_A']

def validate_task(task,state):
 if task.get('kind') not in ['P0','P1','P2'] or task['N']!=state.N:raise ValueError('task category/length')
 for p in task.get('targets',[]):
  if not 0<=p['index']<state.N or p['tolerance_A']<=0 or np.asarray(p['position']).shape!=(3,) or not np.isfinite(p['position']).all():raise ValueError('target position/support')
  if p.get('orientation') is not None:
   m=np.asarray(p['orientation']);tol=p['orientation_tolerance_deg']
   if m.shape!=(3,3) or not np.allclose(m@m.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(m),1,atol=1e-6) or p['index']<2 or not 0<tol<=180:raise ValueError('right handed target frame')
 for i,a in task.get('motifs',{}).items():
  if not 0<=int(i)<state.N or a not in AA:raise ValueError('sequence constraint')

class World:
 def __init__(self,state,task,limits,seed=1,gamma=.995,shaping=.1,operational_events=4096):
  validate_task(task,state)
  if state.t<3 or not state.valid.all():raise ValueError('construction requires three genuine valid starting anchors')
  self.state=state;self.task=copy.deepcopy(task);self.limits=limits;self.seed=seed;self.rng=torch.Generator(device=state.ca.device).manual_seed(seed);self.gamma=gamma;self.shaping=shaping;self.safeguard=operational_events;self.events=[];self.snapshots={};self.history=[];self.interventions=[];self.branch=0;self.context_branch=0;self.pending=False;self.terminated=False;self.truncated=False;self.snapshot()
 def snapshot(self):self.snapshots[self.state.t]=copy.deepcopy((self.state,self.rng.get_state(),self.history,self.context_branch))
 def action_mask(self,for_bootstrap=False):
  m=torch.zeros(9,dtype=torch.bool,device=self.state.ca.device)
  if not(self.terminated or (self.truncated and not for_bootstrap)):
   m[:min(4,max(1,len(self.task.get('targets',[]))))]=self.state.t<self.state.N and not self.pending
   for i,k in enumerate([4,8,16]):m[4+i]=self.state.t-k>=3 and self.state.t-k in self.snapshots and not self.pending
   m[7]=self.pending;m[8]=True
  return m
 def observe(self):return dict(sequence=''.join(self.state.residues),coordinates=self.state.ca.detach().cpu().tolist(),valid=self.state.valid.cpu().tolist(),goal=copy.deepcopy(self.task),intervention_results=copy.deepcopy(self.interventions),pending=self.pending,branch=self.context_branch)
 def emit(self,event,old,success=0):
  new=0. if self.terminated else potential(self.state,self.task);event.update(reward=success+self.shaping*(self.gamma*new-old),terminal_success=success,terminated=self.terminated);self.events.append(event)
  if len(self.events)>=self.safeguard and not self.terminated:self.truncated=True
 def step(self,a,g,aa_logp=None,geometry_logp=None):
  if not self.action_mask()[0]:raise ValueError('construction not available')
  old=potential(self.state,self.task);t=self.state.t;mask=mask_for_step(t);self.state=append(self.state,BuildAction(AA[int(a)],g,mask));event=dict(kind='RESIDUE',a=int(a),g=g.detach().cpu().tolist(),mask=list(mask),aa_logp=aa_logp,geometry_logp=geometry_logp,branch=self.context_branch);self.history.append(event);self.emit(event,old);self.snapshot()
 def option(self,index,continue_fn=None,steps=1):
  if not 0<=index<9 or not self.action_mask()[index]:raise ValueError('masked high action')
  if steps<1:raise ValueError('no zero-duration options')
  start=len(self.events);old=potential(self.state,self.task)
  if index<4:
   if continue_fn is None:raise ValueError('continuation policy required')
   for _ in range(steps):
    if self.state.t==self.state.N or self.truncated:break
    continue_fn(self,index)
  elif index<7:
   old_t=self.state.t;old_tip=self.state.ca[-1].clone();k=self.state.t-[4,8,16][index-4];self.state,rng,self.history,self.context_branch=copy.deepcopy(self.snapshots[k]);self.rng.set_state(rng);self.snapshots={i:v for i,v in self.snapshots.items() if i<=k};self.pending=True;self.interventions.append(dict(kind='REWIND',from_t=old_t,to=k,endpoint_displacement_A=float(torch.linalg.vector_norm(old_tip-self.state.ca[-1]))));self.emit(dict(kind='REWIND',to=k),old)
  elif index==7:
   self.branch+=1;self.context_branch=self.branch;self.rng.manual_seed(self.seed+1000003*self.branch);self.pending=False;self.interventions.append(dict(kind='REGENERATE',branch=self.branch));self.emit(dict(kind='REGENERATE',branch=self.branch),old);self.snapshot()
  else:self.terminated=True;self.emit(dict(kind='STOP'),old,float(assess(self.state,self.task,self.limits)['success']))
  rewards=[e['reward'] for e in self.events[start:]]
  if not rewards:raise RuntimeError('empty high transition')
  return dict(reward=sum(self.gamma**i*r for i,r in enumerate(rewards)),duration=len(rewards),discount=self.gamma**len(rewards),terminated=self.terminated,truncated=self.truncated)
 def export(self):return copy.deepcopy(self.__dict__ | {'rng':self.rng.get_state()})
 def restore(self,s):
  rng=s['rng'];self.__dict__.update(copy.deepcopy({k:v for k,v in s.items() if k!='rng'}));self.rng=torch.Generator(device=self.state.ca.device);self.rng.set_state(rng)

def reconstruct(initial,events,task,limits,seed):
 s=BuildState(tuple(initial['sequence']),torch.tensor(initial['coordinates'],dtype=torch.float64),initial['N'],torch.tensor(initial['valid'],dtype=torch.bool));w=World(s,task,limits,initial.get('seed',seed),operational_events=initial.get('operational_events',4096))
 for e in events:
  if e['kind']=='RESIDUE':
   if e.get('rng_draws'):
    torch.rand((),generator=w.rng);torch.randn(3,dtype=torch.float64,generator=w.rng)
   w.step(e['a'],torch.tensor(e['g'],dtype=torch.float64),e.get('aa_logp'),e.get('geometry_logp'))
  elif e['kind']=='REWIND':
   delta=w.state.t-e['to']
   if delta not in [4,8,16]:raise ValueError('unknown rewind option')
   w.option(4+[4,8,16].index(delta))
  elif e['kind']=='REGENERATE':w.option(7)
  elif e['kind']=='STOP':w.option(8)
  else:raise ValueError('unknown event')
 return w

def geometric_jacobian(state,g):
 """Exact local next-Cα derivative only; no ungenerated suffix derivative claimed."""
 def fn(z):return next_ca(state.ca,z)
 return torch.autograd.functional.jacobian(fn,g.double().requires_grad_(True))
