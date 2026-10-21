"""E017 V2: frozen J1 output adapters and serializable hierarchical suffix construction.
No production RL trainer; no arbitrary internal edits or causal KV cache.
"""
from crisp_v2_common import *
import copy,math,time
from torch import nn
from torch.nn import functional as F
from protein_joint_sac.contracts import AA,BuildState,BuildAction
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.models.geometry_inference import corrected_action
from protein_joint_sac.geometry.ca_builder import append,mask_for_step
from partial_finalize import metrics

def context(state,task,subgoal=0,interventions=0):
 x=state.ca.detach().cpu().numpy();F0=frame(x);remaining=max(1,state.N-state.t);goal=np.array(task['endpoint']);target=goal.copy()
 if subgoal==1 and task['waypoint']and state.t<=task['waypoint']['index']:target=np.array(task['waypoint']['position'])
 residual=(goal-x[-1])@F0.T/(3.8*remaining);rot=np.eye(3)if task['orientation']is None else np.array(task['orientation'])@F0.T
 aa=np.zeros(20);motif=task['motifs'].get(str(state.t));
 if motif:aa[AA.index(motif)]=1
 future=[int(k)-state.t for k in task['motifs']if int(k)>=state.t];until=min(future)/state.N if future else 1.
 z=np.r_[state.N/100,state.t/state.N,task['kind']=='P1',task['kind']=='P2',residual,rot.ravel(),aa,until,float(motif is not None),(target-x[-1])@F0.T/(3.8*remaining),min(interventions,10)/10]
 assert z.shape==(42,)and np.isfinite(z).all();return torch.tensor(z,dtype=torch.float32,device=state.ca.device)

class Steering(nn.Module):
 def __init__(self):
  super().__init__();self.A=nn.Sequential(nn.Linear(394,64),nn.SiLU(),nn.Linear(64,20));self.G=nn.Sequential(nn.Linear(414,64),nn.SiLU(),nn.Linear(64,6))
  for net in [self.A,self.G]:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
 def forward(self,h,c,logits,mu,ls,aa=True,geometry=True):
  if h.ndim==1:h=h[None];c=c[None];logits=logits[None];mu=mu[None];ls=ls[None];single=True
  else:single=False
  da=2*torch.tanh(self.A(torch.cat([h.float(),c],-1)))if aa else torch.zeros_like(logits)
  ids=torch.eye(20,device=h.device)[None].expand(len(h),20,20);hc=torch.cat([h.float(),c],-1)[:,None].expand(len(h),20,394)
  dg=torch.tanh(self.G(torch.cat([hc,ids],-1)))if geometry else torch.zeros(len(h),20,6,device=h.device)
  scale=dg.new_tensor([.05,.35,.5]);pm=mu+dg[:,:,:3]*scale;pl=(ls+dg[:,:,3:]*.2).clamp(-5,2);out=(logits.float()+da,pm,pl)
  return tuple(v[0]for v in out)if single else out

def kl(logits,mu,ls,ref_logits,ref_mu,ref_ls):
 lp=logits.log_softmax(-1);rp=ref_logits.log_softmax(-1);p=lp.exp();a=(p*(lp-rp)).sum(-1)
 g=ref_ls-ls+.5*((ls.exp()/ref_ls.exp()).square()+((mu-ref_mu)/ref_ls.exp()).square())-.5
 # All current tasks start at3, so all3internal components are defined.
 conditional=g.sum(-1);return a,(p*conditional).sum(-1),g

@torch.no_grad()
def reference(actor,states):
 from protein_joint_sac.models.batched_bc import collate,encode_batch
 batch=collate(states,'cuda')
 with torch.autocast('cuda',dtype=torch.bfloat16):
  h=encode_batch(actor,batch);la=actor.aa_head(h);B=len(states);ids=torch.arange(20,device='cuda')[None].expand(B,20);hh=h[:,None].expand(B,20,352);chem=torch.zeros(B,20,22,device='cuda');out=actor.geom(torch.cat([hh,actor.selected_aa(ids),chem],-1))
 return h.float(),la.float(),out[:,:,:3].float(),out[:,:,3:].float().clamp(-5,2)

def state_for(task,device='cuda'):
 return BuildState(tuple(task['initial_sequence']),torch.tensor(task['initial_CA'],dtype=torch.float64,device=device),task['N'])

def assess(state,task,limits):
 complete=state.t==state.N
 if not complete:return dict(complete=False,success=False,reason='incomplete',position_error_A=None,orientation_error_deg=None)
 x=state.ca.detach().cpu().numpy();m=metrics(x);pe=float(np.linalg.norm(x[-1]-np.array(task['endpoint'])));oe=None if task['orientation']is None else math.degrees(math.acos(float(np.clip((np.trace(frame(x)@np.array(task['orientation']).T)-1)/2,-1,1))))
 we=None if task['waypoint']is None else float(np.linalg.norm(x[task['waypoint']['index']]-np.array(task['waypoint']['position'])))
 motif=all(state.residues[int(i)]==a for i,a in task['motifs'].items());d=np.linalg.norm(np.diff(x,axis=0),axis=1)
 valid=bool(np.isfinite(x).all()and d.min()>=limits['adjacent_low_A']and d.max()<=limits['adjacent_high_A']and m['clashes_per100_residues']<=limits['clashes_per100_max']and limits['Rg_normalized_low']<=m['Rg_over_N_cuberoot']<=limits['Rg_normalized_high']and m['degenerate_frames']==0)
 return dict(complete=True,success=bool(motif and pe<=task['position_tolerance_A']and(oe is None or oe<=task['orientation_tolerance_deg'])and(we is None or we<=task['waypoint_tolerance_A'])and valid),motifs=motif,valid=valid,position_error_A=pe,orientation_error_deg=oe,waypoint_error_A=we,metrics=m)

def potential(state,task):
 if state.t==state.N:return 0.
 err=float(torch.linalg.vector_norm(state.ca[-1]-state.ca.new_tensor(task['endpoint'])));return -min(1.,err/(3.8*max(1,state.N-state.t)))
def shaping(old,new,gamma=.995,coefficient=.1):return coefficient*(gamma*new-old)

class Hierarchy:
 actions=('CONTINUE','REWIND','REGENERATE','STOP')
 def __init__(self,actor,adapter,task,seed,aa=True,geometry=True):
  self.actor=actor;self.adapter=adapter;self.task=copy.deepcopy(task);self.seed=seed;self.aa=aa;self.geometry=geometry;self.state=state_for(task);self.rng=torch.Generator(device='cuda').manual_seed(seed);self.ticks=0;self.branch=0;self.context_branch=0;self.reward_events=[];self.history=[];self.interventions=[];self.snapshots={};self.pending=False;self.ended=False;self.calls=0;self.snapshot()
 def snapshot(self):self.snapshots[self.state.t]=dict(state=BuildState(self.state.residues,self.state.ca.clone(),self.state.N,self.state.valid.clone()),rng=self.rng.get_state().clone(),history=copy.deepcopy(self.history),context_branch=self.context_branch)
 @torch.no_grad()
 def continue_one(self,subgoal=0):
  if self.ended or self.state.t>=self.state.N:raise ValueError('no action after terminal')
  cpu=BuildState(self.state.residues,self.state.ca.cpu(),self.state.N,self.state.valid.cpu());h,l,m,s=reference(self.actor,[cpu]);c=context(self.state,self.task,subgoal,self.context_branch);l,m,s=self.adapter(h[0],c,l[0],m[0],s[0],self.aa,self.geometry);self.calls+=1
  u=torch.rand((),device='cuda',generator=self.rng);idx=int(torch.searchsorted(l.softmax(-1).cumsum(-1),u).clamp_max(19));eps=torch.randn(3,device='cuda',dtype=torch.float64,generator=self.rng);g,info=corrected_action(m[idx],s[idx],mask_for_step(self.state.t),eps);old_phi=potential(self.state,self.task);self.state=append(self.state,BuildAction(AA[idx],g,mask_for_step(self.state.t)));self.ticks+=1
  self.history.append(dict(t=self.state.t-1,residue=AA[idx],geometry=g.cpu().tolist(),mask=list(mask_for_step(self.state.t-1)),AA_logp=float(l.log_softmax(-1)[idx]),geometry_log_density_representative=info['represented_action_log_density'],geometry_log_density_latent=info['latent_action_log_density'],support_repairs=info['adjusted_components'],subgoal=subgoal,shaping=shaping(old_phi,potential(self.state,self.task)),branch=self.context_branch));self.emit_reward('RESIDUE',old_phi,potential(self.state,self.task));self.snapshot()
 def act(self,name,**kw):
  if name not in self.actions or self.ended:raise ValueError('closed high-level action')
  old_phi=potential(self.state,self.task)
  if name=='CONTINUE':
   for _ in range(kw.get('steps',1)):
    if self.state.t==self.state.N:break
    self.continue_one(kw.get('subgoal',0))
  elif name=='REWIND':
   k=int(kw['position'])
   if not 3<=k<self.state.t or k not in self.snapshots:raise ValueError('valid earlier prefix checkpoint required')
   old=self.state.t;q=self.snapshots[k];self.state=BuildState(q['state'].residues,q['state'].ca.clone(),q['state'].N,q['state'].valid.clone());self.rng.set_state(q['rng']);self.history=copy.deepcopy(q['history']);self.context_branch=q['context_branch'];self.snapshots={t:v for t,v in self.snapshots.items()if t<=k};self.interventions.append(dict(action='REWIND',from_t=old,to_t=k,tick=self.ticks));self.ticks+=1;self.pending=True;self.emit_reward('REWIND',old_phi,potential(self.state,self.task))
  elif name=='REGENERATE':
   if not self.pending:raise ValueError('REGENERATE follows REWIND')
   self.branch+=1;self.context_branch=self.branch;self.rng.manual_seed(self.seed+1000003*self.branch);self.pending=False;self.interventions.append(dict(action='REGENERATE',branch=self.branch,tick=self.ticks));self.ticks+=1;self.emit_reward('REGENERATE',old_phi,potential(self.state,self.task));self.snapshot()
  else:
   self.ended=True;self.ticks+=1;success=assess(self.state,self.task,read('configs/NATIVE_CALIBRATION.json'))['success'];self.emit_reward('STOP',old_phi,0.,float(success))
 def emit_reward(self,kind,before,after,terminal_success=0.):
  self.reward_events.append(dict(tick=self.ticks-1,kind=kind,before_potential=before,after_potential=after,terminal_success=terminal_success,shaping=shaping(before,after),reward=terminal_success+shaping(before,after)))
 def export(self):
  return dict(aa=self.aa,geometry=self.geometry,adapter_state=copy.deepcopy(self.adapter.state_dict()),reference_sha256=MAIN_SHA,context_branch=self.context_branch,reward_events=self.reward_events,state=self.state,rng=self.rng.get_state(),ticks=self.ticks,branch=self.branch,history=self.history,interventions=self.interventions,snapshots=self.snapshots,pending=self.pending,ended=self.ended,calls=self.calls,seed=self.seed,task=self.task)
 def restore(self,d):
  for k,v in d.items():
   if k=='rng':self.rng.set_state(v)
   elif k=='adapter_state':self.adapter.load_state_dict(v)
   elif k=='reference_sha256':
    if v!=MAIN_SHA:raise ValueError('reference checkpoint mismatch')
   else:setattr(self,k,v)


def option_target(rewards,bootstrap=0.,terminated=False,gamma=.995):
 """Finite-horizon semi-MDP Bellman target. Administrative truncation is not termination."""
 return sum(gamma**i*float(r)for i,r in enumerate(rewards))+(0. if terminated else gamma**len(rewards)*bootstrap)


def necessary_position_reachability(state,task,limits):
 """Necessary contour-length bound for reaching the permitted endpoint BALL, not a sufficiency claim."""
 remaining=state.N-state.t;distance=float(torch.linalg.vector_norm(state.ca[-1]-state.ca.new_tensor(task['endpoint'])))
 return distance<=remaining*limits['adjacent_high_A']+task['position_tolerance_A']
